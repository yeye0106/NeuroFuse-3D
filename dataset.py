import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ADNIMultiModalDataset(Dataset):
    def __init__(self, df, image_dir, clinical_scaler=None, is_train=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train = is_train

        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        self.clinical_cols = ['Age', 'Sex_encoded', 'PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDRSB']
        self.clinical_data = self.df[self.clinical_cols].values

        if clinical_scaler is not None:
            self.clinical_data = clinical_scaler.transform(self.clinical_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = str(row['Image Data ID'])
        subject = str(row['Subject'])
        img_name = f"{subject}_{img_id}.nii.gz"
        img_path = os.path.join(self.image_dir, img_name)

        img_data = nib.load(img_path).get_fdata()

        # ==========================================
        # 优化点 1: 3D 在线数据增强 (仅限训练集)
        # ==========================================
        if self.is_train:
            # 50% 概率左右镜像翻转 (Sagittal 轴)，人类左右脑本就基本对称
            if random.random() > 0.5:
                img_data = np.flip(img_data, axis=0).copy()

            # 50% 概率加入微小的随机高斯噪声，防止模型对像素过度敏感
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_data.shape)
                img_data = img_data + noise
                img_data = np.clip(img_data, 0, 1)  # 保持归一化边界

        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
        clinical_tensor = torch.tensor(self.clinical_data[idx], dtype=torch.float32)

        label_str = row['Group']
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)

        return img_tensor, clinical_tensor, label


def get_dataloaders(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data']['csv_path'])
    image_dir = config['data']['image_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    seed = config['data']['seed']
    train_r, val_r, test_r = config['data']['split_ratio']

    subjects = df['Subject'].unique()
    train_subs, temp_subs = train_test_split(subjects, test_size=(val_r + test_r), random_state=seed)
    val_subs, test_subs = train_test_split(temp_subs, test_size=(test_r / (val_r + test_r)), random_state=seed)

    train_df = df[df['Subject'].isin(train_subs)]
    val_df = df[df['Subject'].isin(val_subs)]
    test_df = df[df['Subject'].isin(test_subs)]

    clinical_cols = ['Age', 'Sex_encoded', 'PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDRSB']
    scaler = StandardScaler()
    scaler.fit(train_df[clinical_cols].values)

    train_dataset = ADNIMultiModalDataset(train_df, image_dir, scaler, is_train=True)
    val_dataset = ADNIMultiModalDataset(val_df, image_dir, scaler, is_train=False)
    test_dataset = ADNIMultiModalDataset(test_df, image_dir, scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader