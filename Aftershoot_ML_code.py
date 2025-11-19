import os
import glob
import random
import math
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration

TRAIN_PATH = r"C:/Users/jagad/Downloads/14648881b93c11f0/dataset/Train"
VALID_PATH = r"C:/Users/jagad/Downloads/14648881b93c11f0/dataset/Validation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

META_MLP_OUT = 128
SEED = 42

# Utilities

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


def find_image_file(img_dir, img_id):
    """Find image file with any valid extension"""
    for ext in [".tiff", ".tif", ".png", ".jpg", ".jpeg"]:
        fp = os.path.join(img_dir, f"{img_id}{ext}")
        if os.path.exists(fp):
            return fp

    files = glob.glob(os.path.join(img_dir, f"{img_id}.*"))
    return files[0] if files else None


# Dataset

class WBHybridDataset(Dataset):
    def __init__(self, df, images_dir, meta_cols, target_cols=None,
                 transform=None, scaler=None, label_encoders=None, fit_encoders=False):

        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.meta_cols = meta_cols
        self.target_cols = target_cols
        self.transform = transform

        # Split meta cols
        self.num_cols = [c for c in meta_cols if np.issubdtype(df[c].dtype, np.number)]
        self.cat_cols = [c for c in meta_cols if c not in self.num_cols]

        # Fit label encoders
        self.label_encoders = label_encoders or {}
        if fit_encoders:
            for col in self.cat_cols:
                le = LabelEncoder()
                le.fit(self.df[col].fillna("___nan___").astype(str))
                self.label_encoders[col] = le

        # Encode categorical
        for col in self.cat_cols:
            le = self.label_encoders[col]
            vals = self.df[col].fillna("___nan___").astype(str)
            mapped = vals.apply(lambda x: le.transform([x])[0]
                                if x in le.classes_ else len(le.classes_))
            self.df[f"_cat_{col}"] = mapped

        # Scale numeric
        self.scaler = scaler
        if fit_encoders:
            self.scaler = StandardScaler()
            self.df[self.num_cols] = self.df[self.num_cols].fillna(self.df[self.num_cols].median())
            self.scaler.fit(self.df[self.num_cols])

        if self.num_cols:
            self.df["_meta_num"] = list(self.scaler.transform(self.df[self.num_cols]))
        else:
            self.df["_meta_num"] = [np.array([])] * len(self.df)

        if self.cat_cols:
            self.df["_meta_cat"] = list(self.df[[f"_cat_{c}" for c in self.cat_cols]].values)
        else:
            self.df["_meta_cat"] = [np.array([])] * len(self.df)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_global"]

        img_path = find_image_file(self.images_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        meta = np.concatenate([
            np.array(row["_meta_num"], dtype=np.float32),
            np.array(row["_meta_cat"], dtype=np.float32),
        ])

        if self.target_cols:
            target = row[self.target_cols].values.astype(np.float32)
            return img, torch.tensor(meta), torch.tensor(target), img_id

        return img, torch.tensor(meta), img_id


# Model

class HybridWBModel(nn.Module):
    def __init__(self, meta_dim):
        super().__init__()

        # Image backbone
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        backbone_out = 512

        # Metadata MLP
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, META_MLP_OUT),
            nn.ReLU(),
            nn.Linear(META_MLP_OUT, 64),
            nn.ReLU(),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(backbone_out + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, meta):
        x = self.backbone(img)
        m = self.meta_net(meta)
        out = torch.cat([x, m], dim=1)
        return self.head(out)



# Evaluation

def evaluate(model, loader, criterion):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for img, meta, target, _ in loader:
            img, meta, target = img.to(DEVICE), meta.to(DEVICE), target.to(DEVICE)
            out = model(img, meta)
            preds.append(out.cpu().numpy())
            trues.append(target.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    mae = np.mean(np.abs(preds - trues), axis=0)
    return mae, mae.mean()



# MAIN

if __name__ == "__main__":

    # Load CSV
    train_csv = pd.read_csv(os.path.join(TRAIN_PATH, "sliders.csv"))
    target_cols = ["Temperature", "Tint"]

    # Remove touchTime safely
    meta_cols = [c for c in train_csv.columns if c not in ["id_global", "Temperature", "Tint", "touchTime"]]

    num_cols = [c for c in meta_cols if np.issubdtype(train_csv[c].dtype, np.number)]
    cat_cols = [c for c in meta_cols if c not in num_cols]

    # Fit encoders
    scaler = StandardScaler()
    train_csv[num_cols] = train_csv[num_cols].fillna(train_csv[num_cols].median())
    scaler.fit(train_csv[num_cols])

    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(train_csv[c].fillna("___nan___").astype(str))
        label_encoders[c] = le

    # Split train-validation
    train_df, val_df = train_test_split(train_csv, test_size=0.12, random_state=SEED)

    # Image transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_ds = WBHybridDataset(
        train_df, os.path.join(TRAIN_PATH, "images"),
        meta_cols, target_cols,
        train_tf, scaler, label_encoders, fit_encoders=False
    )

    val_ds = WBHybridDataset(
        val_df, os.path.join(TRAIN_PATH, "images"),
        meta_cols, target_cols,
        val_tf, scaler, label_encoders, fit_encoders=False
    )

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    meta_dim = len(num_cols) + len(cat_cols)
    model = HybridWBModel(meta_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    # Training
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for img, meta, target, _ in pbar:
            img, meta, target = img.to(DEVICE), meta.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            out = model(img, meta)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        mae_t, mae_avg = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch} → MAE: {mae_avg:.4f} (Temp={mae_t[0]:.1f}, Tint={mae_t[1]:.1f})")

    # Save model
    torch.save(model.state_dict(), "hybrid_model_final.pt")


    # Inference
    
    val_inputs = pd.read_csv(os.path.join(VALID_PATH, "sliders_input.csv"))
    val_images = os.path.join(VALID_PATH, "images")

    infer_ds = WBHybridDataset(
        val_inputs, val_images, meta_cols,
        target_cols=None, transform=val_tf,
        scaler=scaler, label_encoders=label_encoders, fit_encoders=False
    )

    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    preds, ids = [], []

    with torch.no_grad():
        for img, meta, id_batch in tqdm(infer_loader, desc="Predicting"):
            img, meta = img.to(DEVICE), meta.to(DEVICE)
            out = model(img, meta).cpu().numpy()
            preds.append(out)
            ids.extend(id_batch)

    preds = np.vstack(preds)
    preds = np.rint(preds).astype(int)

    submission = pd.DataFrame({
        "id_global": ids,
        "Temperature": preds[:, 0],
        "Tint": preds[:, 1],
    })

    submission = submission.set_index("id_global").loc[val_inputs["id_global"]].reset_index()
    submission.to_csv("submission_hybrid.csv", index=False)

    print("Saved: submission_hybrid.csv")

