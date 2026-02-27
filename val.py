import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.measure import label
import segmentation_models_pytorch as smp
from models.models import DualStreamUNet_CBAM, DualStreamUNet_MS
from AJI import calculate_aji
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train import MonuSegDataset


test_path = r"data/kmms_test"
baseline_pth_path = r"results_monuseg_2/Baseline_DiceOnly/best.pth"
ours_pth_path = r"results_monuseg_2/DualStream_MS_MixedLoss/best.pth"

CONFIG = {
        "encoder": "resnet50",
        "weights": "imagenet",
        "batch_size": 4,
        "epochs": 100,
        "lr": 1e-4,
        "image_size": 256,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_path": "data/kmms_training",
        "results_root": "./results_comparison",
        "workers": 0,
    }
def get_loader():
    test_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_dataset = MonuSegDataset(
        root_dir=test_path,
        phase='test',
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["workers"],
        pin_memory=True
    )
    return test_loader

def run_evaluation(configs, test_loader, device):
    all_results = []

    for config in configs:
        name = config['name']
        model = config['model'].to(device)
        model.load_state_dict(torch.load(config['path'], map_location=device))
        model.eval()

        print(f"\nVerification in progress: {name} ...")
        m_list = {k: [] for k in ['dice', 'iou', 'precision', 'recall', 'aji']}

        with torch.no_grad():
            for imgs, grays, masks in tqdm(test_loader):
                imgs, grays, masks = imgs.to(device), grays.to(device), masks.to(device)

                outputs = model(imgs, grays) if "DualStream" in name else model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                preds_np = preds.squeeze().cpu().numpy()
                masks_np = masks.squeeze().cpu().numpy()

                if preds_np.ndim == 2:  # bs=1
                    preds_list, masks_list = [preds_np], [masks_np]
                else:
                    preds_list, masks_list = preds_np, masks_np

                for p, m in zip(preds_list, masks_list):
                    tp = np.sum(p * m)
                    fp = np.sum(p * (1 - m))
                    fn = np.sum((1 - p) * m)
                    eps = 1e-7

                    m_list['dice'].append((2 * tp + eps) / (2 * tp + fp + fn + eps))
                    m_list['iou'].append((tp + eps) / (tp + fp + fn + eps))
                    m_list['precision'].append((tp + eps) / (tp + fp + eps))
                    m_list['recall'].append((tp + eps) / (tp + fn + eps))
                    m_list['aji'].append(calculate_aji(m, p))

        avg_res = {k: np.mean(v) for k, v in m_list.items()}
        avg_res['Model'] = name
        all_results.append(avg_res)

    return pd.DataFrame(all_results)


def plot_comparison(df):
    metrics = ['dice', 'iou', 'precision', 'recall', 'aji']
    df_plot = df.set_index('Model')[metrics].T

    ax = df_plot.plot(kind='bar', figsize=(12, 7), rot=0, width=0.8, color=['#95a5a6', '#3498db'])
    plt.title('Performance Comparison: Baseline vs Improved (DualStream)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("\nThe comparison image has been saved as: comparison_result.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        {
            'name': 'Baseline_DiceOnly',
            'model': smp.Unet(encoder_name="resnet50", classes=1),
            'path': baseline_pth_path
        },
        {
            'name': 'DualStream_MS_MixedLoss',
            'model': DualStreamUNet_MS(encoder_name="resnet50"),
            'path': ours_pth_path
        }
    ]

    test_loader = get_loader()
    result_df = run_evaluation(configs, test_loader, device)

    print("\nSummary of comparative experiments：")
    print(result_df[['Model', 'dice', 'iou', 'precision', 'recall', 'aji']])

    plot_comparison(result_df)