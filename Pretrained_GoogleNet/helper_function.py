

# short side 128, crop, Google

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np

class SpectralRatioDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Only keep necessary columns
        self.labels_df = self.labels_df[['filename', 'log_sr_r_avg', 'log_sr_g_avg', 'log_sr_b_avg']]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image file name and path
        img_name = self.labels_df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image file {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        if h < w:
            new_h = 128
            new_w = int(w * (128 / h))
        else:
            new_w = 128
            new_h = int(h * (128 / w))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Scale pixel values appropriately
        if image.dtype == np.uint16:
            image = image.astype('float32') / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype('float32') / 255.0
        else:
            raise TypeError(f"Unsupported image data type: {image.dtype}")

        # Convert image to PyTorch tensor and reorder dimensions to [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Extract labels
        x = self.labels_df.iloc[idx]['log_sr_r_avg']
        y = self.labels_df.iloc[idx]['log_sr_g_avg']
        z = self.labels_df.iloc[idx]['log_sr_b_avg']

        # Convert to float
        x = float(x)
        y = float(y)
        z = float(z)

        # Compute the magnitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        if magnitude == 0:
            raise ValueError(f"Zero magnitude vector for index {idx}")

        # Normalize the vector components
        x_u = x / magnitude
        y_u = y / magnitude
        z_u = z / magnitude

        # Compute elevation angle (el)
        elevation = np.arccos(z_u)  # el = arccos(z_u)

        # Compute azimuth angle (az)
        azimuth = np.arctan2(y_u, x_u)  # az = arctan2(y_u, x_u)

        labels = np.array([azimuth, elevation], dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels
        


