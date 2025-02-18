import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report


# Below are some functions to compute visual attention metrics.
# See reference: https://github.com/MemoonaTahira/Visual-Saliency-Metrics-for-Evaluating-Deep-Learning-Model-performance/blob/main/_visual_attention_metrics.py
def kld(pred, gt, eps=1e-8):
    """KL-divergence. It treats pred/gt as a distribution.
    pred and gt are heatmaps of size batch_size x h x w.
    """
    pred_sum = torch.sum(pred, dim=[1, 2], keepdim=True)
    gt_sum = torch.sum(gt, dim=[1, 2], keepdim=True)
    pred = pred / (pred_sum + eps)
    gt = gt / (gt_sum + eps)

    kl = torch.sum(gt * torch.log(gt / (pred + eps) + eps), dim=[1, 2])
    return kl


def mse_loss(pred, gt):
    m = torch.mean((pred - gt) ** 2, dim=[1, 2])
    # m = torch.sqrt(m)
    return m


def cc(pred, gt):
    """Pearson correlation coefficient"""
    pred_mean = torch.mean(pred, dim=[1, 2], keepdim=True)
    gt_mean = torch.mean(gt, dim=[1, 2], keepdim=True)

    numerator = torch.sum(
        (pred - pred_mean) * (gt - gt_mean), dim=[1, 2]
    )
    den_pred = torch.sqrt(
        torch.sum((pred - pred_mean) ** 2, dim=[1, 2])
    )
    den_gt = torch.sqrt(
        torch.sum((gt - gt_mean) ** 2, dim=[1, 2])
    )

    cc_ = numerator / (den_pred * den_gt + 1e-8)
    return cc_


def nss(pred, gt, eps=1e-8):
    """NSS (Normalized Scanpath Saliency). See https://arxiv.org/pdf/1604.03605"""
    pred_mean = torch.mean(pred, dim=[1, 2], keepdim=True)
    pred_std = torch.std(pred, dim=[1, 2], unbiased=True, keepdim=True)
    pred = (pred - pred_mean) / (pred_std + eps)
    pred.masked_fill_(gt <= 0, 0)
    nss_ = pred.sum(dim=[1, 2]) / (gt > 0).sum(dim=[1, 2])
    return nss_


def similarity(pred, gt, eps=1e-8):
    """Similiarity metric"""
    pred_sum = torch.sum(pred, dim=[1, 2], keepdim=True)
    gt_sum = torch.sum(gt, dim=[1, 2], keepdim=True)
    pred = pred / (pred_sum + eps)
    gt = gt / (gt_sum + eps)
    sim = torch.sum(torch.minimum(pred, gt), dim=[1, 2])
    return sim


def AUC_Judd_single(saliencyMap, fixationMap, jitter=True):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        # print('Error: no fixationMap')
        score = float('nan')
        return score

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)

    return score


def AUC_Judd(pred, gt):
    """AUC Judd (Area Under the ROC Curve, Judd version)"""
    res = []
    for p, g in zip(torch.unbind(pred, 0), torch.unbind(gt, 0)):
        score = AUC_Judd_single(p.cpu().numpy(), g.cpu().numpy())
        res.append(score)
    return torch.tensor(res)


# Below are metrics to compare misalignment text sequences

def text_alignment_single(pred_text, gt_text):
    """Compute metrics for misalignment text sequences.
    pred_text and gt_text should be of the form: "a yellow_0 cat"
    """

    pred = pred_text.strip().split()
    gt = gt_text.strip().split()

    pred_labels = []
    gt_labels = []
    
    if len(pred) < len(gt):
        pred = pred + [''] * (len(gt) - len(pred))

    for p, g in zip(pred, gt):
        if g.endswith('_0'):
            gt_labels.append(1)
            g = g[:-2]
        else:
            gt_labels.append(0)
        
        if p.endswith('_0'):
            if p[:-2] == g:
                pred_labels.append(1)
            else:
                pred_labels.append(-1)
        else:
            if p == g:
                pred_labels.append(0)
            else:
                pred_labels.append(-1)

    rep = classification_report(gt_labels, pred_labels, labels=[0, 1], output_dict=True, zero_division=0)

    out = {}
    for key, val in rep.items():
        key = key.replace(' ', '_')
        if key == 'accuracy':
            out[f'seq/{key}'] = val
        else:
            for c, v in val.items():
                out[f'seq/{c}/{key}'] = v

    return out


def text_alignment(pred_texts, gt_texts):
    results = defaultdict(list)
    for pred, gt in zip(pred_texts, gt_texts):
        rep = text_alignment_single(pred, gt)
        for k, v in rep.items():
            results[k].append(v)
    
    out = {}
    for k, v in results.items():
        out[k] = np.mean(v)
    return out


metrics_func = {
    'kld': kld,
    'nss': nss,
    'cc': cc,
    'similarity': similarity,
    'AUC_Judd': AUC_Judd,
}


if __name__ == "__main__":
    gt = ["a yellow cat"]
    pred = ["a yellow_0 cat"]
    print(text_alignment(pred, gt))