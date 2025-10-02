import os, sys, json, math, random, time
from pathlib import Path
from typing import Tuple, Dict, Any, List
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

# # For evaluation metrics mAP, IoU and Dice
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from torchmetrics.classification import JaccardIndex
# from torchmetrics.segmentation import DiceScore
from eval_test import evaluate_test # Customized functions

import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

import timm
from tqdm import tqdm


# ----------------------------
# Example config file
# ----------------------------
# TRAIN_JSON = "/content/drive/MyDrive/DL_cell_segmentation/data/train.json"
# TEST_JSON   = "/content/drive/MyDrive/DL_cell_segmentation/data/test.json"
# IM_ROOT    = Path("/content/drive/MyDrive/DL_cell_segmentation/data/train/original").parent  # will infer per json

# OUT_DIR    = Path("/content/drive/MyDrive/DL_cell_segmentation/outputs/vitb_maskrcnn")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# NUM_EPOCHS = 2            # 10 for quick run, but 20 might be better
# BATCH_SIZE = 2
# LR         = 2e-4
# IMG_SIZE   = 800           # 1024 resize shorter side to this (keeps aspect)
# FREEZE_VIT = False         # set True to freeze backbone at first
# USE_VIT_L  = False         # False=ViT-B/16, True=ViT-L/16
# DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# LOSS_FILE  = '/content/drive/MyDrive/DL_cell_segmentation/outputs/loss.txt'
# EVAL_FILE  = '/content/drive/MyDrive/DL_cell_segmentation/outputs/eval.txt'

def load_config(fn):
  '''
  Load cofiguration file
  fn: file name to the config file
  '''
  dict_config = {}
  with open(fn) as fh:
    for line in fh:
      if line!= '':
        var_name, val = line.split('=')
        dict_config[var_name.strip()] = val.strip()
  return dict_config

config_fn = sys.argv[1] # Get congifurations
dict_config = load_config(config_fn)

# Load config to global variables
TRAIN_JSON = dict_config['TRAIN_JSON']
TEST_JSON = dict_config['TEST_JSON']
IM_ROOT = dict_config['IM_ROOT']
OUT_DIR    = Path(dict_config['OUT_DIR'])
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_EPOCHS = int(dict_config['NUM_EPOCHS']) # 10 for quick run, but 20 might be better
BATCH_SIZE = int(dict_config['BATCH_SIZE'])
LR         = float(dict_config['LR'])
IMG_SIZE   = int(dict_config['IMG_SIZE']) # 1024 resize shorter side to this (keeps aspect)
FREEZE_VIT = bool(dict_config['FREEZE_VIT']) # set True to freeze backbone at first
USE_VIT_L  = bool(dict_config['USE_VIT_L']) # False=ViT-B/16, True=ViT-L/16
DEVICE     = eval(dict_config['DEVICE'])
LOSS_FILE  =  dict_config['LOSS_FILE']
EVAL_FILE  = dict_config['EVAL_FILE']

loss_fh = open(LOSS_FILE, 'w') # Output loss over batch and epoch
eval_fh = open(EVAL_FILE, 'w') # Output IoU, mAP, Dice on test set
# Write header line
loss_fh.write(f"epoch\ttrain_avg\tval_avg\n")
eval_fh.write(f"epoch\tIoU\tmAP\tDice\n")


# ----------------------------
# Utils
# ----------------------------

def coco_root_from_json(json_path: str) -> Path:
    # Try to locate the image directory by inspecting the first image entry.
    coco = COCO(json_path)
    img_info = coco.loadImgs(coco.getImgIds()[0])[0]
    fp = img_info["file_name"]
    # If json has absolute paths you can skip this.
    # Otherwise try siblings: train/original, train/images, images
    candidates = [
        Path(fp).parent,
        Path(json_path).parent / "train/original",
        Path(json_path).parent / "test/original",
        Path(json_path).parent
    ]
    for c in candidates:
        if (c / Path(fp).name).exists():
            return c
    # Fallback: ask user to adjust IM_ROOT
    return Path(json_path).parent


def rle_to_mask(seg: Dict[str, Any]) -> np.ndarray:
    """Decode COCO RLE to binary mask."""
    rle = seg
    if isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
    m = mask_utils.decode(rle)  # (H, W, N) or (H, W)
    if m.ndim == 3:
        m = np.any(m, axis=-1)
    return m.astype(np.uint8)


# ----------------------------
# Dataset
# ----------------------------
class CocoInstanceDataset(Dataset):
    def __init__(self, json_path: str, img_dir: Path = None, aug: bool = False):
        self.coco = COCO(json_path)
        self.img_ids = self.coco.getImgIds()
        self.img_dir = img_dir if img_dir is not None else coco_root_from_json(json_path)
        self.aug = aug

        # Albumentations pipeline (keeps masks aligned)
        if aug:
            self.tf = A.Compose([
                A.LongestMaxSize(IMG_SIZE),
                A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]))
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(IMG_SIZE),
                A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]))

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / Path(img_info["file_name"]).name

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        boxes = []
        labels = []
        areas = []

        for a in anns:
            if "segmentation" not in a: 
                continue
            seg = a["segmentation"]
            if isinstance(seg, dict) and "counts" in seg:
                m = rle_to_mask(seg)
            else:
                # If polygons, convert to RLE then mask
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(rles)
                if m.ndim == 3:
                    m = np.any(m, axis=-1).astype(np.uint8)
            if m.sum() == 0:
                continue

            # bbox in COCO is [x, y, w, h]
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            masks.append(m)
            labels.append(a["category_id"])
            areas.append(a.get("area", float(bw*bh)))

        if len(boxes) == 0:
            # Torchvision Mask R-CNN wants at least one target; create a dummy small background box
            boxes = [[0, 0, 1, 1]]
            masks = [np.zeros((h, w), dtype=np.uint8)]
            labels = [0]  # background id (won't be used if your categories start at 1)
            areas = [1.0]

        masks = np.stack(masks, 0)  # (N, H, W)

        # Albumentations expects pascal_voc boxes
        bboxes = boxes.copy()
        bbox_labels = labels.copy()
        auged = self.tf(image=img, masks=list(masks), bboxes=bboxes, bbox_labels=bbox_labels)
        img = auged["image"]
        masks = np.stack(auged["masks"], 0) if len(auged["masks"]) > 0 else np.zeros_like(masks)
        bboxes = auged["bboxes"]
        labels = auged["bbox_labels"]

        # Convert to tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # normalize like typical ViT (timm default IMAGENET_IN1K)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_t = (img_t - mean) / std

        boxes_t = torch.as_tensor(bboxes, dtype=torch.float32) if len(bboxes)>0 else torch.zeros((0,4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        masks_t = torch.from_numpy(masks.astype(np.uint8))

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas[:len(labels)], dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------------------
# ViT backbone + tiny FPN
# ----------------------------
class ViTBackboneWithFPN(nn.Module):
    """
    Wrap a timm ViT to provide a dict of pyramid features for Mask R-CNN.
    Strategy:
      - take final ViT feature map (B, C, H/16, W/16),
      - make a 4-level pyramid: P2(1/4), P3(1/8), P4(1/16), P5(1/32)
      - P2/P3 are created by upsampling the ViT feature; P5 by downsampling.
    This is simple but works reasonably as a baseline.
    """
    def __init__(self, vit_name="vit_base_patch16_224", img_size=IMG_SIZE):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=(3,), img_size=img_size)
        self.out_channels = 256

        vit_ch = self.vit.feature_info.channels()[-1]  # e.g., 768 for ViT-B, 1024 for ViT-L
        # project to a common channel width
        self.proj = nn.Conv2d(vit_ch, self.out_channels, kernel_size=1)

        # prepare P2,P3,P4,P5 from single stride-16 map
        self.p5_down = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.p3_up   = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=2, stride=2)
        self.p2_up   = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=4, stride=4)

        # add an FPN layer to smooth
        self.fpn = FeaturePyramidNetwork(in_channels_list=[self.out_channels]*4,
                                         out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.vit(x)  # list with single map at stride 16
        f16 = self.proj(feats[-1])  # (B, 256, H/16, W/16)

        f4  = self.p2_up(f16)  # P2
        f8  = self.p3_up(f16)  # P3
        f16 = f16              # P4
        f32 = self.p5_down(f16)  # P5

        feats_dict = {"0": f4, "1": f8, "2": f16, "3": f32}  # keys can be any strings
        out = self.fpn(feats_dict)
        return out


def build_model(vit_large=False) -> MaskRCNN:
    '''
    Build R-CNN model
    '''
    vit_name = "vit_large_patch16_224" if vit_large else "vit_base_patch16_224"
    backbone = ViTBackboneWithFPN(vit_name=vit_name, img_size=IMG_SIZE)
   
    # For debug: check how many feature maps this backbone returns
    with torch.no_grad():
        feats = backbone(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    print("FPN levels and shapes:", {k: v.shape for k,v in feats.items()})
    
    # 1) One feature map => sizes/aspect_ratios must be length 1 (tuple-of-tuples)
    # anchor_generator = AnchorGenerator(
    #     sizes=((32, 64, 128, 256),),      # anchors for the single map
    #     aspect_ratios=((0.5, 1.0, 2.0),), # same: length 1
    # )
    anchor_sizes     = ((16,), (32,), (64,), (128,))      # four levels
    aspect_ratios    = ((0.5, 1.0, 2.0),) * 4             # repeat 4 times
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )
    
    # 2) RPN head needs to know how many anchors per location (here 3 ratios)
    num_anchors_per_loc = anchor_generator.num_anchors_per_location()[0]
    rpn_head = RPNHead(backbone.out_channels, num_anchors_per_loc)
    
    # 3) Build Mask R-CNN with fixed image size that matches your ViT
    model = MaskRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        min_size=IMG_SIZE,
        max_size=IMG_SIZE,
        fixed_size=(IMG_SIZE, IMG_SIZE),   # if your torchvision supports it
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    return model


# ----------------------------
# Train / Eval
# ----------------------------
def evaluate(model, dl, device):
    '''
    - model
    - dl: dataloader
    '''
    was_training = model.training
    model.train()  # <-- force train mode so Mask R-CNN returns loss dict

    totals = defaultdict(float)
    n = 0

    with torch.enable_grad():  # don't use torch.no_grad() here
        for images, targets in dl:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)  # dict of losses
            loss = sum(loss_dict.values())

            totals['loss'] += loss.item()
            for k, v in loss_dict.items():
                totals[k] += v.item()
            n += 1

    # restore previous mode
    if not was_training:
        model.eval()

    # averages
    return {k: v / max(n, 1) for k, v in totals.items()}


def train():
    # datasets
    train_dir = coco_root_from_json(TRAIN_JSON)
    test_dir   = coco_root_from_json(TEST_JSON)

    # ##### Split training data into train (70%) + validation (30%) #####
    ds_full = CocoInstanceDataset(TRAIN_JSON, img_dir=train_dir, aug=True)
    # Shuffle indices
    indices = list(range(len(ds_full)))
    random.shuffle(indices)

    # Split train:validation = 70%:30%
    split = int(0.7 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = Subset(ds_full, train_idx)
    val_dataset   = Subset(ds_full, val_idx)

    # Build loaders
    dl_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)
    dl_val = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
    
    test_dataset = CocoInstanceDataset(TEST_JSON,   img_dir=test_dir,   aug=False)
    dl_test = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
    
    model = build_model(vit_large=USE_VIT_L).to(DEVICE)

    if FREEZE_VIT:
        for p in model.backbone.vit.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=0.05)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val = math.inf
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(dl_train, desc=f"Train {epoch+1}/{NUM_EPOCHS}"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        lr_sched.step()
        train_avg = epoch_loss / max(1, len(dl_train))
        val_results = evaluate(model, dl_val, DEVICE)
        val_avg = val_results["loss"]   # pick total loss

        print(f"# Epoch {epoch+1}: train_loss={train_avg:.4f}  val_loss={val_avg:.4f}")
        loss_fh.write(f"{epoch+1}\t{train_avg:.4f}\t{val_avg:.4f}\n")

        # Apply evaluations on test set to check performance
        test_results = evaluate_test(model, dl_test, DEVICE)
        # print(test_results)
        eval_fh.write(f"{epoch+1}\t{test_results['IoU']:.4f}\t{test_results['mAP']:.4f}\t{test_results['Dice']:.4f}\n")


        # Save checkpoint for each epoch (may take a lot of storage)
        is_best = val_avg < best_val
        best_val = min(best_val, val_avg)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_avg,
        }
        torch.save(ckpt, OUT_DIR / f"checkpoint_{epoch+1:03d}.pth")
        if is_best:
            torch.save(ckpt, OUT_DIR / "best.pth")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()


loss_fh.close()
eval_fh.close()
