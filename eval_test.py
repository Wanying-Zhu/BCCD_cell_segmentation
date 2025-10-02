'''
Evaluate model performance in test set using IoU, Dice, and mAP

'''

# For evaluation metrics mAP, IoU and Dice
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import JaccardIndex # IoU
from torchmetrics.segmentation import DiceScore

import torch


# ##### Helpers for mAP #####
def _preds_for_map(outputs, score_thresh=0.0):
    """
    Convert Mask R-CNN outputs to torchmetrics MeanAveragePrecision format.
    outputs: list of dicts from torchvision (boxes, labels, scores, masks)
    Returns: list[dict(boxes, scores, labels, masks)]
    """
    preds = []
    for o in outputs:
        keep = o["scores"] >= score_thresh
        d = {
            "boxes":  o["boxes"][keep].detach().cpu(),
            "scores": o["scores"][keep].detach().cpu(),
            "labels": o["labels"][keep].detach().cpu(),
        }
        if "masks" in o:
            # (N,1,H,W) -> (N,H,W), bool
            d["masks"] = (o["masks"][keep].detach().cpu() > 0.5).squeeze(1)
        preds.append(d)
    return preds

def _targets_for_map(targets):
    """
    Convert your dataset targets to torchmetrics format.
    Each target is already a dict with boxes, labels, masks (torch tensors).
    """
    tgts = []
    for t in targets:
        d = {
            "boxes":  t["boxes"].detach().cpu(),
            "labels": t["labels"].detach().cpu(),
        }
        if "masks" in t:
            d["masks"] = (t["masks"].detach().cpu() > 0.5)
        tgts.append(d)
    return tgts


# ##### Helpers for IoU and Dice #####
def _union_mask_from_instances(masks, H, W):
    """
    masks: (N,H,W) boolean or (N,1,H,W) float/bool
    Returns a single (H,W) boolean mask = union of instances.
    """
    if masks is None or masks.numel() == 0:
        return torch.zeros((H, W), dtype=torch.bool)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    masks = masks > 0.5
    return masks.any(dim=0)



@torch.no_grad()
def evaluate_test(model, dl_test, device):
    model.eval()

    # TorchMetrics: binary IoU & Dice (semantic), and mAP for instances
    iou_metric  = JaccardIndex(task="binary").to(device)  # IoU
    dice_metric = DiceScore(num_classes=2).to(device) # Dice (Assuming 2 classes: background and cell)
    map_metric  = MeanAveragePrecision(iou_type="segm")   # stays on CPU

    for images, targets in dl_test:
        images = list(img.to(device) for img in images)

        # forward
        outputs = model(images)

        # === mAP (instance) ===
        preds_map  = _preds_for_map(outputs, score_thresh=0.0)  # keep all, mAP handles thresholds
        targets_map = _targets_for_map(targets)
        map_metric.update(preds_map, targets_map)

        # === IoU & Dice (semantic) ===
        for out, tgt in zip(outputs, targets):
            # Get H,W from the original image size saved by torchvision wrapper
            # If you don't have it, infer from target masks (preferred) or image tensor
            if "masks" in tgt and tgt["masks"].numel() > 0:
                H, W = tgt["masks"].shape[-2:]
            else:
                # fall back to output mask size
                if "masks" in out and out["masks"].numel() > 0:
                    H, W = out["masks"].shape[-2:]
                else:
                    # skip if no masks anywhere
                    continue

            # union masks
            pred_union = _union_mask_from_instances(
                out.get("masks", torch.empty(0, 1, H, W, device=device)).to(device),
                H, W
            )
            gt_union = _union_mask_from_instances(
                tgt.get("masks", torch.empty(0, H, W, device=device)).to(device),
                H, W
            )

            # to float for torchmetrics (binary task accepts bool or float)
            pred_union = pred_union.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            gt_union   = gt_union.float().unsqueeze(0).unsqueeze(0)    # (1,1,H,W)

            iou_metric.update(pred_union, gt_union)
            dice_metric.update(pred_union, gt_union)

    # aggregate
    iou  = iou_metric.compute().item()
    dice = dice_metric.compute().item()
    map_res = map_metric.compute()  # dict with map, map_50, map_75, etc.

    # For a compact summary:
    out = {
        "IoU":  iou,
        "Dice": dice,
        "mAP":  float(map_res.get("map", torch.tensor(float("nan")))),
        "mAP_50": float(map_res.get("map_50", torch.tensor(float("nan")))),
        "mAP_75": float(map_res.get("map_75", torch.tensor(float("nan")))),
    }
    return out


