import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class ISICSkinDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        csv_file   : processed CSV (train or val)
        image_dir  : directory containing raw images
        transform  : torchvision transforms
        """
        nb_dir = Path.cwd()
        project_root = nb_dir if (nb_dir / 'data').exists() else nb_dir.parent
        csv_file = pd.read_csv(str(project_root / 'data' / 'processed' / 'train' / 'train_binary.csv'))
        image_dir = Path(project_root / 'data' / 'raw' / 'train' / 'images_train')

        self.df = csv_file
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Get one row from processed CSV
        row = self.df.iloc[idx]

        # Image ID from CSV
        image_id = row["isic_id"]

        # Label (already binary: 0 or 1)
        label = torch.tensor(row["label"], dtype=torch.long)

        # Construct image path
        image_path = f"{self.image_dir}/{image_id}.jpg"

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)

        return image, label

