import os
import tqdm
import numpy as np

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr  # 기본 1e-4
    weight_decay = 1e-5      # AdamW 최적값

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------
    # Model load
    # ------------------------------
    net = resnet50()

    if args.pre_weights is not None:
        pattern = "yolov1_([0-9]+)"
        strs = args.pre_weights.split(".")[-2]
        f_name = strs.split("/")[-1]
        epoch_str = re.search(pattern, f_name).group(1)
        epoch_start = int(epoch_str) + 1
        net.load_state_dict(torch.load(f"./weights/{args.pre_weights}")["state_dict"])
    else:
        epoch_start = 1
        res = torchvision.models.resnet50(pretrained=True)
        new_state = res.state_dict()
        net_dict = net.state_dict()

        for k in new_state.keys():
            if k in net_dict.keys() and not k.startswith("fc"):
                net_dict[k] = new_state[k]
        net.load_state_dict(net_dict)

    net = net.to(device)
    criterion = yoloLoss().to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    # ========================================
    # 🔥 AdamW Optimizer + Stable setting
    # ========================================
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # ========================================
    # 🔥 Warmup + Cosine Annealing Scheduler
    # ========================================
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        # warmup 끝난 후 cosine decay
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------
    # Dataset Loading
    # ------------------------------------------
    with open("./Dataset/train.txt") as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    with open("./Dataset/test.txt") as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=os.cpu_count()
    )

    print(f"NUMBER OF DATA SAMPLES: {len(train_dataset)}")
    print(f"BATCH SIZE: {batch_size}")

    # ============================================
    # Training Loop
    # ============================================
    for epoch in range(epoch_start, num_epochs + 1):

        net.train()
        total_loss = 0.0

        print(("\n" + "%10s" * 3) % ("epoch", "loss", "gpu"))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device).float()

            pred = net(images)

            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mem = (
                "%.3gG"
                % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            )
            s = ("%10s" + "%10.4g" + "%10s") % (
                "%g/%g" % (epoch, num_epochs),
                total_loss / (i + 1),
                mem,
            )
            progress_bar.set_description(s)

        # 🔥 Scheduler update
        scheduler.step()

        # ============================================
        # Validation
        # ============================================
        validation_loss = 0.0
        net.eval()

        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device).float()

                prediction = net(images)
                loss = criterion(prediction, target)

                validation_loss += loss.item()

        validation_loss /= len(test_loader)
        print(f"Validation_Loss: {validation_loss:07.3}")

        # weight save
        if epoch % 10 == 0:
            save = {"state_dict": net.state_dict()}
            torch.save(save, f"./weights/yolov1_adamw_{epoch:04d}.pth")

    save = {"state_dict": net.state_dict()}
    torch.save(save, "./weights/yolov1_adamw_final.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)  # 안정화 버전 기본값 변경
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--pre_weights", type=str)
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)

    args = parser.parse_args()
    main(args)
