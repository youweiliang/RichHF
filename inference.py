import os
import argparse
import json
import importlib.util
import sys
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from train import evaluate_model, HuggingFaceDataset


# Config
parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default=None, help="The path where trained model is saved.")
parser.add_argument("--ckpt", type=str, default=None, help="Or directly use the path to the trained model.")
# parser.add_argument("--vit_model", type=str, default="google/vit-large-patch16-384", help="Name of the Vision Transformer model.")
# parser.add_argument("--t5_model", type=str, default="t5-base", help="Name of the T5 model.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size in inference.")
parser.add_argument("--best", action="store_true", help="Whether to use the best model by val metrics.")
parser.add_argument("--infer", action="store_true", help="Do inference and visualization of heatmaps.")
parser.add_argument("--eval", action="store_true", help="Do evaluation and calculation of metrics.")
args = parser.parse_args()
print(args)


def import_model_from_path(script_path):
    """Load the model config from the log_dir"""
    module_name = "rahf_model"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RAHF

if args.log_dir is not None:
    RAHF = import_model_from_path(os.path.join(args.log_dir, 'model.py'))
    with open(os.path.join(args.log_dir, 'args_config.json')) as f:
        config = json.load(f)
else:
    from model import RAHF
    config = {'vit_model': 'google/vit-large-patch16-384', 't5_model': 't5-base', 'multi_heads': True}

vit_model = config['vit_model']

patch_size = 16
image_size = 384
if vit_model.startswith('google/vit'):
    patch_size = int(vit_model.split('-')[-2].replace('patch', ''))
    image_size = int(vit_model.split('-')[-1])

model = RAHF(
    vit_model=config['vit_model'], t5_model=config['t5_model'], 
    multi_heads=config['multi_heads'], patch_size=patch_size, image_size=image_size
)


def infer(model, dataloader, device, log_dir, tag, max_iter=None):
    model.eval()

    data = defaultdict(list)

    iter_ = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch['image'] = batch['image'].to(device)
            outputs = model(batch['image'], batch['caption'])

            data['pred_text'].extend(outputs['output_seq'])

            data['gt_text'].extend(batch['target_text'])
            data['image'].append(batch['image'].cpu())
            data['caption'].extend(batch['caption'])

            out_scores = outputs['scores']
            for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
                data[f'gt_score_{score}'].append(batch[score])
                data[f'pred_score_{score}'].append(out_scores[score].cpu())

            out_heatmaps = outputs['heatmaps']
            for heatmap in ('implausibility', 'misalignment'):
                data[f'gt_heatmap_{heatmap}'].append(batch[heatmap])
                data[f'pred_heatmap_{heatmap}'].append(out_heatmaps[heatmap].cpu())

            iter_ += 1
            if max_iter is not None and iter_ >= max_iter:
                break

    out_js = []

    for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
        data[f'gt_score_{score}'] = torch.cat(data[f'gt_score_{score}']).flatten().tolist()
        data[f'pred_score_{score}'] = torch.cat(data[f'pred_score_{score}']).flatten().tolist()

    for i in range(len(data['caption'])):
        tmp = {'idx': i}
        for k in ['caption', 'gt_text', 'pred_text']:
            tmp[k] = data[k][i]
        for score in ('plausibility', 'alignment', 'aesthetics', 'overall'):
            tmp[f'gt_score_{score}'] = data[f'gt_score_{score}'][i]
            tmp[f'pred_score_{score}'] = data[f'pred_score_{score}'][i]

        out_js.append(tmp)

    if args.best:
        tag += '_best'

    with open(os.path.join(log_dir, f'{tag}_outputs.json'), 'w') as f:
        json.dump(out_js, f, indent=2)

    data['image'] = torch.cat(data['image']) * 0.5 + 0.5  # denormalize (works for google's ViTs)

    for heatmap in ('implausibility', 'misalignment'):
        data[f'gt_heatmap_{heatmap}'] = torch.cat(data[f'gt_heatmap_{heatmap}']).unsqueeze(1).expand_as(data['image'])
        data[f'pred_heatmap_{heatmap}'] = torch.cat(data[f'pred_heatmap_{heatmap}']).unsqueeze(1).expand_as(data['image'])

    img_path = os.path.join(log_dir, f'{tag}_imgs')
    os.makedirs(img_path, exist_ok=True)

    for i in range(len(data['caption'])):
        imgs1 = [data['image'][i]]
        imgs2 = [data['image'][i]]
        for heatmap in ('implausibility', 'misalignment'):
            imgs1.append(data[f'gt_heatmap_{heatmap}'][i])
            imgs2.append(data[f'pred_heatmap_{heatmap}'][i])
        imgs1 = torch.cat(imgs1, dim=2)
        imgs2 = torch.cat(imgs2, dim=2)
        imgs = torch.cat([imgs1, imgs2], dim=1)
        imgs = imgs.permute([1, 2, 0])  # change to h x w x 3
        imgs = imgs * 255
        im = Image.fromarray(imgs.numpy().astype('uint8'))
        im.save(os.path.join(img_path, f'{i:03d}.jpg'))

# Load dataset
dataset_path = './data/rich_human_feedback_dataset'
full_dataset = load_from_disk(dataset_path)


# Create PyTorch datasets and dataloaders
train_dataset = HuggingFaceDataset(full_dataset["train"], vit_model, image_size)
dev_dataset = HuggingFaceDataset(full_dataset["dev"], vit_model, image_size)
test_dataset = HuggingFaceDataset(full_dataset["test"], vit_model, image_size)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)#, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)#, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)#, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.log_dir is not None:
    ckpt_name = 'model_best.pt' if args.best else 'model.pt'
    ckpt = torch.load(os.path.join(args.log_dir, ckpt_name), map_location='cpu')
else:
    ckpt = torch.load(args.ckpt, map_location='cpu')
msg = model.load_state_dict(ckpt)
print("Loaded model:", msg)
model = model.to(device)


if args.eval:
    metrics = evaluate_model(model, test_dataloader, nn.MSELoss(), device, 'test')
    file = f"test_metrics{'_best' if args.best else ''}.json"
    with open(os.path.join(args.log_dir, file), 'wt') as f:
        json.dump(metrics, f, indent=4)

if args.infer:
    infer(model, test_dataloader, device, args.log_dir, 'test')
    # infer(model, dev_dataloader, device, args.log_dir, 'dev')
    # infer(model, train_dataloader, device, args.log_dir, 'train', max_iter=10)