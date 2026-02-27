import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.models import DualStreamUNet_MS, DualStreamUNet_CBAM
import matplotlib.pyplot as plt
from AJI import batch_aji

# ==========================================
#        1. Global Configuration
# ==========================================

CONFIG = {
    "encoder": "resnet50",
    "weights": None,
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-4,
    "image_size": 256,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_path": "data/kmms_training",
    "results_root": "./results_monuseg",
    "workers":0,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================================
#        2. Data augmentation of 24 training images
# ========================================================

def get_transforms(phase="train", size=256):
    if phase == "train":
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.OpticalDistortion(distort_limit=0.2, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ], p=0.4),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            # ---------------------
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ==========================================
#        3. Dataset loading
# ==========================================

class MonuSegDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.image_ids = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.tif', '.jpg', '.png'))])

        split_idx = int(len(self.image_ids) * 0.8)
        self.image_ids = self.image_ids[:split_idx] if phase == "train" else self.image_ids[split_idx:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        image = cv2.imread(os.path.join(self.images_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))

        mask_name = img_name.replace(".jpg", ".png").replace(".tif", ".png")
        mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Generate CLAHE grayscale branches
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_gray = torch.from_numpy(clahe.apply(gray)).unsqueeze(0).float() / 255.0

        return image, image_gray, mask.unsqueeze(0).float()


# ===================================================================
#        4. Training engine (with loss switching functionality)
# ===================================================================

def train_engine(model, train_loader, val_loader, experiment_name, use_bce=False):
    save_dir = os.path.join(CONFIG["results_root"], experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    device = CONFIG['device']
    model = model.to(device)

    # Loss definition
    criterion_dice = smp.losses.DiceLoss(mode="binary")
    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_dice = 0.0
    history = []

    print(f"\n>>> Running: {experiment_name} (Use_BCE={use_bce})")

    for epoch in range(CONFIG['epochs']):
        # --- train ---
        model.train()
        train_loss = 0
        for imgs, grays, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=False):
            imgs, grays, masks = imgs.to(device), grays.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(imgs, grays) if "DualStream" in experiment_name else model(imgs)

            # Loss Strategy Switch
            if use_bce:
                loss = 0.5 * criterion_dice(outputs, masks) + 0.5 * criterion_bce(outputs, masks)
            else:
                loss = criterion_dice(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- val ---
        model.eval()
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_aji = 0.0

        with torch.no_grad():
            for imgs, grays, masks in val_loader:
                imgs, grays, masks = imgs.to(device), grays.to(device), masks.to(device)
                outputs = model(imgs, grays) if "DualStream" in experiment_name else model(imgs)

                # 1. calculate TP, FP, FN, TN (for Precision, Recall, IoU, Dice)
                tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.long(), mode='binary', threshold=0.5)
                total_tp += tp.sum().item()
                total_fp += fp.sum().item()
                total_fn += fn.sum().item()
                total_tn += tn.sum().item()

                # 2. calculate AJI (use batch_aji)
                # batch_aji returns the sum of the AJI values of all images in that batch.
                batch_aji_score = batch_aji(outputs, masks)
                total_aji += batch_aji_score

        # --- Indicator Calculation ---
        eps = 1e-7
        dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
        iou = total_tp / (total_tp + total_fp + total_fn + eps)
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        # Average AJI
        avg_aji = total_aji / len(val_loader.dataset)

        # logs
        record = {
            "Epoch": epoch + 1,
            "Loss": train_loss / len(train_loader),
            "Dice": dice,
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "AJI": avg_aji
        }
        history.append(record)

        scheduler.step()

        # save model weight (Base on Dice)
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        print(f"Epoch {epoch + 1} | Dice: {dice:.4f} | IoU: {iou:.4f} | AJI: {avg_aji:.4f}")

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "log.csv"), index=False)
    return df


# ================================================
#        5. Visualization plotting functions
# ================================================

def plot_experiment_comparison(all_results, save_path="comparison_metrics.png"):
    """
    Create a graph comparing the metrics of all experimental groups.

    all_results: dict, key=experiment name, value=DataFrame (containing training history)
    """
    metrics = ['Loss', 'Dice', 'IoU', 'Precision', 'Recall', 'AJI']

    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Performance Comparison of Different Models', fontsize=16)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        for j, (exp_name, df) in enumerate(all_results.items()):
            color = colors[j % len(colors)]
            ax.plot(df['Epoch'], df[metric], label=exp_name, color=color, linewidth=1.5)

        ax.set_title(metric)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(CONFIG["results_root"], save_path), dpi=300)
    print(f"\n[Info] Comparison plot saved to {os.path.join(CONFIG['results_root'], save_path)}")
    plt.close()


# =================================================
#        6. Perform a comparative experiment
# =================================================

if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['results_root'], exist_ok=True)

    # data loading
    train_ds = MonuSegDataset(CONFIG['dataset_path'], "train", get_transforms("train"))
    val_ds = MonuSegDataset(CONFIG['dataset_path'], "val", get_transforms("val"))
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=CONFIG['workers'])

    # config
    model_factory = {
        "Baseline": lambda: smp.Unet(encoder_name=CONFIG['encoder'], encoder_weights=CONFIG['weights'], in_channels=3,
                                     classes=1),
        "DualStream_MS": lambda: DualStreamUNet_MS(encoder_name=CONFIG['encoder'], encoder_weights=CONFIG['weights']),
        # "DualStream_CBAM": lambda: DualStreamUNet_CBAM(encoder_name=CONFIG['encoder'],
        #                                                encoder_weights=CONFIG['weights'])
    }

    # A dictionary used to collect all results
    all_experiment_results = {}

    # Phase 1：DiceOnly
    print("\n=== Stage 1: Dice Loss Only ===")
    for name in model_factory:
        set_seed(CONFIG['seed'])  # Ensure that the initial random seed is consistent for each model.
        exp_name = f"{name}_DiceOnly"
        df = train_engine(model_factory[name](), train_loader, val_loader, exp_name, use_bce=False)
        all_experiment_results[exp_name] = df

    # Phase 2：Dice + BCE
    print("\n=== Stage 2: Dice + BCE Loss ===")
    for name in model_factory:
        set_seed(CONFIG['seed'])
        exp_name = f"{name}_MixedLoss"
        df = train_engine(model_factory[name](), train_loader, val_loader, exp_name, use_bce=True)
        all_experiment_results[exp_name] = df

    # plot
    print("\nPlotting results...")
    plot_experiment_comparison(all_experiment_results)

    print("\nAll experiments finished!")