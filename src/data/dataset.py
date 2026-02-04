import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class ISICSkinDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        csv_file  : Path to processed CSV (train_binary.csv or val_binary.csv)
        image_dir : Path to raw images directory
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["isic_id"]
        label = torch.tensor(row["label"], dtype=torch.long)

        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


