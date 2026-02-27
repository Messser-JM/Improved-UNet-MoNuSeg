import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class NucleiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        # Filter out all .tif files and arrange them in order to ensure correspondence.
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. get filename and path
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 2. Replace .tif with .png to find the mask.
        mask_name = img_name.replace('.tif', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 3. Read img
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Read Mask (grayscale mode)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"找不到对应的 Mask 文件: {mask_path}")

        # 5.
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # 6. Normalization and Dimension Transformation
        # Image: [H, W, C] -> [C, H, W]
        image = image.transpose(2, 0, 1).astype('float32') / 255.0
        # Mask: [H, W] -> [1, H, W]
        mask = np.expand_dims(mask, 0).astype('float32') / 255.0

        return torch.tensor(image), torch.tensor(mask)


# --- test ---
if __name__ == "__main__":
    # Enter the path to the folder where you store your data locally.
    # Assuming your project directory is Nuclei_Seg_Project/data/kmms_training
    DATA_PATH = './data/kmms_training'

    try:
        dataset = NucleiDataset(DATA_PATH)
        print(f"The dataset was successfully loaded, containing {len(dataset)} images.")

        img, mask = dataset[0]
        print(f"img Tensor shape: {img.shape}")
        print(f"Mask Tensor shape: {mask.shape}")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image (.tif)")
        plt.imshow(img.permute(1, 2, 0).numpy())

        plt.subplot(1, 2, 2)
        plt.title("Mask (.png)")
        plt.imshow(mask.squeeze().numpy(), cmap='gray')
        plt.show()

    except Exception as e:
        print(f"Runtime error: {e}")
        print("Please check if the filenames of the image and mask files in the folder are completely identical except for the file extension.")