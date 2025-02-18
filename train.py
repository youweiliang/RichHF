import io
import os
import shutil
import datetime
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from model import RAHF
from metrics import metrics_func, mse_loss, text_alignment
from torch.utils.tensorboard import SummaryWriter


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vit_model="google/vit-large-patch16-384", image_size=384):
        self.dataset = hf_dataset
        self.image_transform = AutoImageProcessor.from_pretrained(vit_model)
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def preprocess_heatmap(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        return self.heatmap_transform(image).squeeze(0)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        out = {}
        img = Image.open(io.BytesIO(sample["image"]))
        out['image'] = self.image_transform(img, return_tensors="pt")['pixel_values'][0]

        out['implausibility'] = self.preprocess_heatmap(sample["artifact_map"])
        out['misalignment'] = self.preprocess_heatmap(sample["misalignment_map"])

        out['plausibility'] = np.array(sample['artifact_score'], dtype=np.float32)
        out['alignment'] = np.array(sample['misalignment_score'], dtype=np.float32)
        out['overall'] = np.array(sample['overall_score'], dtype=np.float32)
        out['aesthetics'] = np.array(sample['aesthetics_score'], dtype=np.float32)

        out['caption'] = sample["clean_prompt"]
        out['target_text'] = sample["labeled_prompt"]
        return out


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    target_images = torch.stack([item["target_image"] for item in batch])
    captions = [item["caption"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    scores = torch.stack([item["score"] for item in batch])
    return {"images": images, "target_images": target_images, "captions": captions, "target_texts": target_texts, "scores": scores}


# Training function
def train_model(model, dataloader, optimizer, mse_criterion, device):
    model.train()
    total_loss = []
    iter_ = 0
    loss_all = defaultdict(list)
    for batch in tqdm(dataloader, desc="Training", leave=False):
        for key in list(batch.keys()):
            val = batch[key]
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)

        optimizer.zero_grad()
        outputs = model(batch['image'], batch['caption'], batch['target_text'])

        total_batch_loss = outputs['seq_loss'] * args.seq_weight

        loss_dict = {}
        out_scores = outputs['scores']
        for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
            loss_dict[score] = mse_criterion(batch[score], out_scores[score])
            total_batch_loss += loss_dict[score] * args.score_weight

        out_heatmaps = outputs['heatmaps']
        for heatmap in ('implausibility', 'misalignment'):
            loss_dict[heatmap] = mse_criterion(batch[heatmap], out_heatmaps[heatmap])
            total_batch_loss += loss_dict[heatmap] * args.heatmap_weight

        total_batch_loss.backward()
        optimizer.step()

        for key, val in loss_dict.items():
            loss_all[key].append(val.item())
        loss_all['seq_loss'].append(outputs['seq_loss'].item())

        total_loss.append(total_batch_loss.item())
        iter_ += 1

    for key in list(loss_all.keys()):
        loss_all[key] = np.mean(loss_all[key])

    return np.mean(total_loss), loss_all


def avg_metric(metrics):
    """Compute an overall metric for the rich feedback metrics"""
    s = 0
    for sc in ['plausibility', 'alignment', 'aesthetics', 'overall']:
        for mt in ['plcc', 'srcc']:
            s += metrics[f"score/{mt}/{sc}"]
    
    h = 0
    for hm in ['implausibility', 'misalignment']:
        for mt in ['nss', 'cc', 'similarity', 'AUC_Judd']:
            h += metrics[f"heatmap/{mt}/{hm}"]

    al = metrics["seq/precision/macro_avg"] + metrics["seq/f1-score/1"]

    avg = s / 8 + al / 2 + h / 4
    return avg


def evaluate_model(model, dataloader, mse_criterion, device, phase="Validation"):
    model.eval()

    predicted_scores = defaultdict(list)
    target_scores = defaultdict(list)
    loss_dict = defaultdict(list)

    heatmap_metrics = defaultdict(list)

    gt_texts = []
    pred_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase}", leave=False):
            for key in list(batch.keys()):
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(batch['image'], batch['caption'], batch['target_text'])
            total_batch_loss = outputs['seq_loss'] * args.seq_weight

            outputs2 = model(batch['image'], batch['caption'])
            gt_texts.extend(batch['target_text'])
            pred_texts.extend(outputs2['output_seq'])

            out_scores = outputs['scores']
            for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
                loss_dict[score].append(mse_criterion(batch[score], out_scores[score]).item())
                total_batch_loss += loss_dict[score][-1]  * args.score_weight
            
                predicted_scores[score].append(out_scores[score].cpu().numpy())
                target_scores[score].append(batch[score].cpu().numpy())

            out_heatmaps = outputs['heatmaps']
            for heatmap in ('implausibility', 'misalignment'):
                gt = batch[heatmap]
                pred = out_heatmaps[heatmap]
                loss_dict[heatmap].append(mse_criterion(gt, pred).item())
                total_batch_loss += loss_dict[heatmap][-1] * args.heatmap_weight

                gt_sum = gt.sum(dim=[1, 2])
                nonzero_gt = gt_sum > 0
                if torch.any(~nonzero_gt):
                    mse_l = mse_loss(pred[~nonzero_gt], gt[~nonzero_gt])
                    heatmap_metrics[f'mse/{heatmap}'].append(mse_l)
                if torch.any(nonzero_gt):
                    pred_, gt_ = pred[nonzero_gt], gt[nonzero_gt]
                    for key, func in metrics_func.items():
                        metric_val = func(pred_, gt_)
                        heatmap_metrics[f'{key}/{heatmap}'].append(metric_val)

            loss_dict['seq_loss'].append(outputs['seq_loss'].item())
            loss_dict['total_loss'].append(total_batch_loss.item())

    metrics = {}
    for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
        pred = np.concatenate(predicted_scores[score])
        gt = np.concatenate(target_scores[score])
        metrics[f'score/plcc/{score}'] = pearsonr(pred, gt)[0]
        metrics[f'score/srcc/{score}'] = spearmanr(pred, gt)[0]
        metrics[f'loss/{score}'] = np.mean(loss_dict[score])

    for heatmap in ('implausibility', 'misalignment'):
        metrics[f'loss/{heatmap}'] = np.mean(loss_dict[heatmap])

    for key, vals in heatmap_metrics.items():
        val = torch.cat(vals).mean().item()
        metrics[f"heatmap/{key}"] = val

    metrics['loss/seq'] = np.mean(loss_dict['seq_loss'])
    metrics['loss/val'] = np.mean(loss_dict['total_loss'])

    text_align_metrics = text_alignment(pred_texts, gt_texts)
    metrics = metrics | text_align_metrics
    metrics['avg_metric'] = avg_metric(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and evaluation for RAHF.")

    parser.add_argument("--vit_model", type=str, default="google/vit-large-patch16-384", help="Name of the Vision Transformer model.")
    parser.add_argument("--t5_model", type=str, default="t5-base", help="Name of the T5 model.")
    parser.add_argument("--multi_heads", action="store_true", help="Whether to use multi-heads version or the augmented prompt version.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--init_lr", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate.")
    parser.add_argument("--decoder_lr_scale", type=float, default=1, help="Scale the decoder LR.")
    parser.add_argument("--seq_weight", type=float, default=1, help="Loss weight for seq loss.")
    parser.add_argument("--score_weight", type=float, default=1, help="Loss weight for score loss.")
    parser.add_argument("--heatmap_weight", type=float, default=1, help="Loss weight for heatmap loss.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--freeze_vit", action="store_true", help="Freeze the ViT weights.")
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze the T5 decoder weights.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the checkpoint to resume.")
    parser.add_argument("--log_dir", type=str, default="exp", help="Path (with an additional time stamp) to save model checkpoints and logs.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training (for DDP).")

    args = parser.parse_args()
    print(args)

    # Distributed Setup
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        distributed = True
        # Use the environment variable LOCAL_RANK if available.
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if local_rank == 0:
            print("Distributed training with {} GPUs".format(os.environ["WORLD_SIZE"]))
    else:
        distributed = False
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # Model and Dataset Setup
    vit_model = args.vit_model

    patch_size = None
    image_size = None
    if vit_model.startswith('google/vit'):
        patch_size = int(vit_model.split('-')[-2].replace('patch', ''))
        image_size = int(vit_model.split('-')[-1])
    elif vit_model == 'facebook/dino-vitb8':
        patch_size = 8
        image_size = 224
    elif vit_model == 'facebook/dino-vitb16':
        patch_size = 16
        image_size = 224

    # Load dataset
    dataset_path = './data/rich_human_feedback_dataset'
    full_dataset = load_from_disk(dataset_path)

    # Create PyTorch datasets
    train_dataset = HuggingFaceDataset(full_dataset["train"], args.vit_model, image_size)
    dev_dataset = HuggingFaceDataset(full_dataset["dev"], args.vit_model, image_size)
    test_dataset = HuggingFaceDataset(full_dataset["test"], args.vit_model, image_size)

    # Create dataloaders
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
        # Only rank 0 will run evaluation and logging
        if local_rank == 0:
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        else:
            dev_dataloader = None
            test_dataloader = None
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, optimizer, and loss functions
    model = RAHF(
        vit_model=args.vit_model, t5_model=args.t5_model, 
        multi_heads=args.multi_heads, patch_size=patch_size, image_size=image_size
    )
    if args.ckpt:
        msg = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print(f"Loaded from ckpt:", msg)
    model = model.to(device)

    for n, p in model.named_parameters():
        if 'vit.pooler.dense' in n:  # not used in RAHF
            p.requires_grad = False

    # Wrap model in DDP if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Freeze parameters if requested
    for name, param in model.named_parameters():
        if (name.startswith('t5.decoder') or name.startswith('vit.embeddings')) and args.freeze_decoder:
            param.requires_grad = False
        if name.startswith('vit.encoder') and args.freeze_vit:
            param.requires_grad = False

    num_parameters = 0
    p_other, p_t5_decoder = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if local_rank == 0:
            print(n)
        if "t5.decoder" in n:
            p_t5_decoder.append(p)
        else:
            p_other.append(p)
        num_parameters += p.data.nelement()
    if local_rank == 0:
        print("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {"params": p_other, "lr": args.init_lr},
        {"params": p_t5_decoder, "lr": args.init_lr * args.decoder_lr_scale},
    ]

    optimizer = optim.AdamW(optim_params, lr=args.init_lr, weight_decay=2e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.min_lr)
    mse_criterion = nn.MSELoss()

    # Setup logging (only on rank 0)
    if local_rank == 0:
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.log_dir, tag)
        writer = SummaryWriter(log_dir=log_dir)
        print("log_dir:", log_dir)

        # Save the scripts
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, os.path.join(log_dir, 'train.py'))
        dirname = os.path.dirname(script_path)
        shutil.copy(os.path.join(dirname, 'model.py'), os.path.join(log_dir, 'model.py'))

        # Save the arguments config
        args_dict = vars(args)
        args_dict['log_dir'] = log_dir
        with open(os.path.join(log_dir, "args_config.json"), "w") as json_file:
            json.dump(args_dict, json_file, indent=2)
    else:
        log_dir = None

    # Training loop
    val_metrics_ = []
    max_val_metric = 0
    train_loss_ = []

    for epoch in range(args.num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training phase
        train_loss, loss_all = train_model(model, train_dataloader, optimizer, mse_criterion, device)
        if local_rank == 0:
            print(f"Training Loss: {train_loss:.4f}")
            writer.add_scalar('loss/train', train_loss, epoch)
            for key, val in loss_all.items():
                writer.add_scalar(f'loss/train/{key}', val, epoch)
            train_loss_.append(loss_all)
            with open(os.path.join(log_dir, "train_loss.json"), 'wt') as f:
                json.dump(train_loss_, f, indent=4)

            # Validation phase
            val_metrics = evaluate_model(model, dev_dataloader, mse_criterion, device, phase="Validation")
            print(f"Validation Metrics: {val_metrics}")
            for m, v in val_metrics.items():
                writer.add_scalar(m, v, epoch)
            
            if val_metrics['avg_metric'] > max_val_metric:
                max_val_metric = val_metrics['avg_metric']
                print("Save best checkpoint at epoch", epoch)
                if distributed:
                    torch.save(model.module.state_dict(), os.path.join(log_dir, "model_best.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, "model_best.pt"))

            lr_scheduler.step(epoch)
            lrs = lr_scheduler.get_lr()
            writer.add_scalar("LR", lrs[0], epoch)

            val_metrics['epoch'] = epoch
            val_metrics_.append(val_metrics)
            with open(os.path.join(log_dir, "val_metrics.json"), 'wt') as f:
                json.dump(val_metrics_, f, indent=4)

            if distributed:
                torch.save(model.module.state_dict(), os.path.join(log_dir, "model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))
        else:
            # For non-zero ranks, you may want to step the LR scheduler as well.
            lr_scheduler.step(epoch)

    # Testing phase (only on rank 0)
    if local_rank == 0:
        test_metrics = evaluate_model(model, test_dataloader, mse_criterion, device, phase="Testing")
        print(f"Test Metrics: {test_metrics}")
        with open(os.path.join(log_dir, "test_metrics.json"), 'wt') as f:
            json.dump(test_metrics, f, indent=4)

    if distributed:
        torch.distributed.destroy_process_group()