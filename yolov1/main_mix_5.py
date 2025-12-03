import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_mix_5 import Dataset
from utils.loss_mix_5 import YoloLoss
from utils.ema_mix_5 import ModelEMA
from nets.nn import resnet50 as detnet_resnet

import argparse


# ============================================================
# Warmup + MultiStep LR Scheduler
# ============================================================
class WarmupMultiStepLR:
    def __init__(self, optimizer, warmup_epochs, milestones, gamma, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            lr = self.base_lr
            for m in self.milestones:
                if epoch >= m:
                    lr *= self.gamma

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


# ============================================================
# Main Training
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = detnet_resnet().to(device)
    model.train()

    # EMA
    ema = ModelEMA(model)

    # --------------------------------------------------------
    # Optimizer: SGD Momentum + Nesterov
    # --------------------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=5e-4,
        nesterov=True
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    scheduler = WarmupMultiStepLR(
        optimizer=optimizer,
        warmup_epochs=3,
        milestones=[40, 60],
        gamma=0.1,
        base_lr=args.lr
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    with open("./Dataset/train.txt") as f:
        train_list = f.readlines()

    train_dataset = Dataset(args.data_dir, train_list, train=True, mosaic_prob=0.7, mixup_prob=0.3)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    with open("./Dataset/test.txt") as f:
        val_list = f.readlines()

    val_dataset = Dataset(args.data_dir, val_list, train=False, mosaic_prob=0.0, mixup_prob=0.0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    criterion = YoloLoss().to(device)

    print(f"Training samples: {len(train_dataset)}")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        scheduler.step(epoch)

        total_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            ema.update(model)

            total_loss += loss.item()

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                pred = ema.ema(imgs)
                val_loss += criterion(pred, targets).item()

        val_loss /= len(val_loader)
        print(f"   Validation Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            torch.save({"state_dict": ema.ema.state_dict()},
                       f"./weights/yolov1_mix_5_{epoch:04d}.pth")

    torch.save({"state_dict": ema.ema.state_dict()},
               "./weights/yolov1_mix_5_final.pth")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    args = parser.parse_args()

    main(args)
