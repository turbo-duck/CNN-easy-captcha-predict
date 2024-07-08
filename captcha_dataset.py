import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label_str = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # 转换为整数列表
        label = [int(char) for char in label_str]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

