# main_mix_7.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from nets.nn import resnet50 as detnet_resnet
from utils.dataset_mix_5 import Dataset      # mix_5 기반 dataset 그대로 사용해도 정상 동작
from utils.loss_mix_7 import YoloLoss        # mix_7 전용 loss (CIoU + SmoothL1)
from utils.ema_mix_4 import ModelEMA         # 기존 EMA 그대로 사용

import argparse


# ======================================================
# MIX_7 : 안정 버전
# ======================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Model
    # ------------------------------
    model = detnet_resnet().to(device)
    model.train()

    ema = ModelEMA(model)

    # ------------------------------
    # Optimizer (ResNet50에 가장 안정적)
    # ------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=1e-3,    # ★ 강력한 일반화 효과
        nesterov=True
    )

    # ------------------------------
    # OneCycleLR (YOLOv5/YOLOv8 기본)
    # ------------------------------
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=1,     # epoch 단위 스케줄
        epochs=args.epochs,
        pct_start=0.1,         # 10% 구간에서 LR 최대
        final_div_factor=1000  # 마지막에 LR = LR/1000 까지 감소
    )

    # ------------------------------
    # Dataset & Loader
    # ------------------------------
    with open("./Dataset/train.txt") as f:
        train_names = f.readlines()

    train_dataset = Dataset(
        data_dir=args.data_dir,
        names=train_names,
        train=True,
        mosaic_prob=0.5,
        mixup_prob=0.3
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    with open("./Dataset/test.txt") as f:
        val_names = f.readlines()

    val_dataset = Dataset(
        data_dir=args.data_dir,
        names=val_names,
        train=False,
        mosaic_prob=0.0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")

    # ------------------------------
    # Loss
    # ------------------------------
    criterion = YoloLoss().to(device)

    # ======================================================
    # Training loop
    # ======================================================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            ema.update(model)

            total_loss += loss.item()

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

        # ------------------------------
        # Validation
        # ------------------------------
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

        # ------------------------------
        # Save
        # ------------------------------
        if epoch % 10 == 0:
            os.makedirs("./weights", exist_ok=True)
            torch.save({'state_dict': ema.ema.state_dict()},
                       f"./weights/yolov1_mix_7_{epoch:04d}.pth")

    torch.save({'state_dict': ema.ema.state_dict()},
               "./weights/yolov1_mix_7_final.pth")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    args = parser.parse_args()

    main(args)
