import os
import math
import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from nets.nn import resnet50       # ⭐ 너의 YOLOv1 모델
from utils.loss_mix_6 import YoloLoss    # ⭐ 새 Loss
from utils.dataloader import YOLOv1Dataset   # ⭐ 새 DataLoader


# ==========================================================
# Warmup + Cosine Scheduler
# ==========================================================
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


# ==========================================================
# Basic Transforms
# ==========================================================
def get_transforms():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])


# ==========================================================
# Train One Epoch
# ==========================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0

    for imgs, targets in tqdm.tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


# ==========================================================
# Validation
# ==========================================================
def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, targets in tqdm.tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)
            loss = loss_fn(pred, targets)
            val_loss += loss.item()

    return val_loss / len(loader)


# ==========================================================
# Main
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {device}")

    EPOCHS = 150
    WARMUP = 5
    LR = 1e-4
    BATCH = 8

    # Datasets
    train_dataset = YOLOv1Dataset(
        img_dir="./Dataset/Images",
        label_dir="./Dataset/Labels",
        list_file="./Dataset/train.txt",
        S=14, B=2, C=20,
        transform=get_transforms()
    )

    val_dataset = YOLOv1Dataset(
        img_dir="./Dataset/Images",
        label_dir="./Dataset/Labels",
        list_file="./Dataset/test.txt",
        S=14, B=2, C=20,
        transform=get_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

    # Model
    model = resnet50().to(device)

    # Loss
    loss_fn = YoloLoss(S=14, B=2, C=20)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

    # Scheduler
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        warmup_epochs=WARMUP,
        max_epochs=EPOCHS,
        base_lr=LR,
        min_lr=1e-6
    )

    best_val = 99999999

    # =============================================
    #               TRAIN LOOP
    # =============================================
    for epoch in range(EPOCHS):
        lr = scheduler.step(epoch)
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} | LR: {lr:.6f} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        if (epoch + 1) % 10 == 0:
            val_loss = validate(model, val_loader, loss_fn, device)
            print(f"[E{epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "best_mix_6.pth")
                print(">>> Best model saved!")

        torch.save(model.state_dict(), "last_mix_6.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
