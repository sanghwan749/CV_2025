import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import tqdm

from torch.amp import autocast, GradScaler

from utils.loss_mix_1 import YoloLoss
from utils.dataset import Dataset


# =====================================================
# CBAM Modules
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        maxp = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + maxp)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxp, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, maxp], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


# =====================================================
# Backbone: ResNet101 + CBAM + GAP
# =====================================================
class ResNet101_CBAM(nn.Module):
    def __init__(self, num_classes=20, S=7, B=2):
        super().__init__()

        base = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        layers = list(base.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        self.cbam = CBAM(2048)
        self.conv = nn.Conv2d(2048, 1024, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(1024 * 7 * 7, S * S * (num_classes + B * 5))
        self.S = S
        self.B = B
        self.C = num_classes

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# =====================================================
# Validation function
# =====================================================
def validate(net, criterion, val_loader, device):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = net(imgs)
            loss = criterion(preds, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# =====================================================
# Training
# =====================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ResNet101_CBAM().to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = YoloLoss().to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    # Dataset loading
    with open("./Dataset/train.txt") as f:
        train_names = f.readlines()

    with open("./Dataset/test.txt") as f:
        test_names = f.readlines()

    # transforms must remain a list (Dataset.py compatibility)
    common_transform = [
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]

    train_dataset = Dataset(args.data_dir, train_names, train=True,
                            transform=common_transform)

    test_dataset = Dataset(args.data_dir, test_names, train=False,
                           transform=common_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    warmup_iters = 400

    # =====================================================
    # Training Loop
    # =====================================================
    global_step = 0

    for epoch in range(1, args.epoch + 1):
        net.train()
        pbar = tqdm.tqdm(train_loader)
        step = 0

        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            # ---------- WARMUP LR ----------
            if global_step < warmup_iters:
                warm_lr = args.lr * (global_step / warmup_iters)
                for g in optimizer.param_groups:
                    g["lr"] = warm_lr

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                preds = net(imgs)
                loss = criterion(preds, labels)

            if loss.isnan():
                print("⚠ NaN detected → skipping step")
                optimizer.zero_grad()
                global_step += 1
                continue

            scaler.scale(loss).backward()

            # grad clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f"Epoch[{epoch}/{args.epoch}] Loss:{loss.item():.4f}")

            step += 1
            global_step += 1

        # ----------- VALIDATION EVERY 10 EPOCH ----------- 
        if epoch % 10 == 0:
            val_loss = validate(net, criterion, val_loader, device)
            print(f"[Validation] Epoch {epoch} | Loss: {val_loss:.4f}")

            # Save model every 10 epochs
            torch.save({"state_dict": net.state_dict()},
                       f"./weights/yolov1_mix_1_epoch{epoch}.pth")

    # FINAL SAVE
    torch.save({"state_dict": net.state_dict()}, "./weights/yolov1_mix_1_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    args = parser.parse_args()

    main(args)
