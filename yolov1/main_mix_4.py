import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_mix_4 import Dataset
from utils.loss_mix_4 import YoloLoss
from utils.ema_mix_4 import ModelEMA
from nets.nn import resnet50 as detnet_resnet

import argparse


# ============================================================
# Warmup + CosineAnnealing Learning Rate Scheduler (SAFE VERSION)
# ============================================================
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr

    def step(self, epoch):
        # epoch should start from 1
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
            lr = self.final_lr + (self.base_lr - self.final_lr) * cosine

        # lr is tensor sometimes; convert safely to float
        lr = float(lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ============================================================
# Main Training
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load Model (ResNet50 + DetNet head)
    # --------------------------------------------------------
    model = detnet_resnet()
    model = model.to(device)
    model.train()

    # EMA
    ema = ModelEMA(model)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --------------------------------------------------------
    # LR Scheduler
    # --------------------------------------------------------
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=3,
        total_epochs=args.epochs,
        base_lr=args.lr
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    with open("./Dataset/train.txt") as f:
        train_names = f.readlines()

    train_dataset = Dataset(args.data_dir, train_names, train=True, mosaic_prob=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    with open("./Dataset/test.txt") as f:
        val_names = f.readlines()

    val_dataset = Dataset(args.data_dir, val_names, train=False, mosaic_prob=0.0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    # Loss
    criterion = YoloLoss().to(device)

    print(f"Training samples: {len(train_dataset)}")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        scheduler.step(epoch)

        total_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # -------------------------------
            # Forward (TRAIN model only)
            # -------------------------------
            pred = model(imgs)
            loss = criterion(pred, targets)

            # -------------------------------
            # Backward
            # -------------------------------
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # -------------------------------
            # EMA update (no grad required)
            # -------------------------------
            ema.update(model)

            total_loss += loss.item()

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                # validation은 EMA 모델로 수행
                pred = ema.ema(imgs)
                val_loss += criterion(pred, targets).item()

        val_loss /= len(val_loader)
        print(f"   Validation Loss: {val_loss:.4f}")

        # --------------------------------------------------------
        # Save checkpoint every 10 epochs
        # --------------------------------------------------------
        if epoch % 10 == 0:
            torch.save({"state_dict": ema.ema.state_dict()}, f"./weights/yolov1_mix_4_{epoch:04d}.pth")

    # Final save
    torch.save({"state_dict": ema.ema.state_dict()}, "./weights/yolov1_mix_4_final.pth")
    print("Training complete.")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    args = parser.parse_args()

    main(args)
