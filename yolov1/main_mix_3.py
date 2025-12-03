########################################
# main_mix_3.py
# YOLOv1 + ResNet101 + CBAM + Training
########################################

import os
import re
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms
from utils.dataset import Dataset
from utils.loss_mix_3 import YoloLoss_mix_3


########################################
# YOLOv1 Detection Head
########################################

IMG_SIZE = 448
GRID = 7
B = 2
C = 20


class YOLOv1_Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, GRID * GRID * (C + B * 5))  # 1470 outputs

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)

        out = self.fc2(x)
        out = out.view(-1, GRID, GRID, C + B * 5)
        return out


########################################
# YOLOv1 Model (Backbone + CBAM + Head)
########################################

from main_mix_1 import CBAM    # 네가 사용하던 CBAM 모듈 그대로 import


class YOLO_ResNet_CBAM(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet101(weights=None)
        modules = list(backbone.children())[:-2]  # feature map = (B,2048,14,14)
        self.backbone = nn.Sequential(*modules)

        # 14×14 → 7×7로 줄이는 Pooling 추가
        self.down = nn.AdaptiveAvgPool2d((7, 7))

        self.reduce = nn.Conv2d(2048, 1024, kernel_size=1)
        self.cbam = CBAM(1024)

        self.head = YOLOv1_Head()

    def forward(self, x):
        x = self.backbone(x)     # (B,2048,14,14)
        x = self.down(x)         # (B,2048,7,7) ← 여기서 해결
        x = self.reduce(x)       # (B,1024,7,7)
        x = self.cbam(x)
        out = self.head(x)       # (B,7,7,30)
        return out



########################################
# Training Function
########################################

def train_mix3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr

    # Fix seed
    np.random.seed(42)
    torch.manual_seed(42)

    print("INITIALIZING MODEL...")
    net = YOLO_ResNet_CBAM()

    # Pretrained loading logic
    if args.pre_weights is not None:
        pattern = r"mix3_([0-9]+)"
        fname = args.pre_weights.split('.')[-2]
        f_short = fname.split('/')[-1]
        epoch_str = re.search(pattern, f_short).group(1)
        epoch_start = int(epoch_str) + 1

        net.load_state_dict(torch.load(f'./weights/{args.pre_weights}')["state_dict"])
        print(f"Loaded pretrained weights: {args.pre_weights}")

    else:
        epoch_start = 1
        print("Loading ImageNet pretrained ResNet101 weights...")

        res = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        res_state = res.state_dict()
        net_state = net.state_dict()

        # Load ResNet backbone weights (except fc)
        for k in res_state.keys():
            if k in net_state.keys() and not k.startswith("fc"):
                net_state[k] = res_state[k]

        net.load_state_dict(net_state)
        print("Loaded backbone pretrained weights.")

    # Loss
    criterion = YoloLoss_mix_3().to(device)

    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.train()

    # Optimizer (YOLOv1 style LR group)
    params = []
    for name, p in net.named_parameters():
        if "backbone" in name:
            params.append({"params": p, "lr": lr * 0.1})
        else:
            params.append({"params": p, "lr": lr})

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # Datasets
    with open("./Dataset/train.txt") as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    with open("./Dataset/test.txt") as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False, num_workers=os.cpu_count())

    print(f"TRAIN SAMPLES: {len(train_dataset)}")
    print(f"BATCH SIZE: {batch_size}")

    # Training Loop
    for epoch in range(epoch_start, num_epochs + 1):

        net.train()

        # LR schedule
        if epoch == 30:
            lr = 0.0001
        if epoch == 50:
            lr = 0.00001
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        total_loss = 0.0
        print(("\n" + "%10s" * 3) % ("epoch", "loss", "gpu"))
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, target) in pbar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)

            optimizer.zero_grad()
            loss = criterion(pred, target.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)

            s = ("%10s" + "%10.4g" + "%10s") % (
                f"{epoch}/{num_epochs}",
                total_loss / (i + 1),
                mem
            )
            pbar.set_description(s)

        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in pbar:
                images = images.to(device)
                target = target.to(device)
                pred = net(images)
                loss = criterion(pred, target.float())
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Validation_Loss: {val_loss:07.3}")

        # Save
        if epoch % 10 == 0:
            save = {"state_dict": net.state_dict()}
            torch.save(save, f"./weights/yolov1_mix3_{epoch:04d}.pth")

    # Final save
    save = {"state_dict": net.state_dict()}
    torch.save(save, "./weights/yolov1_mix3_final.pth")


########################################
# Main Entry
########################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--pre_weights", type=str, default=None)

    args = parser.parse_args()

    train_mix3(args)
