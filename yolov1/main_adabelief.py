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
from adabelief_pytorch import AdaBelief


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    # reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------
    # Load model
    # ------------------------------------------
    net = resnet50()

    if args.pre_weights is not None:
        pattern = r'yolov1_([0-9]+)'
        fname = os.path.splitext(os.path.basename(args.pre_weights))[0]
        m = re.search(pattern, fname)
        epoch_start = int(m.group(1)) + 1 if m else 1

        ckpt = torch.load(
            os.path.join('./weights', args.pre_weights),
            map_location='cpu'
        )
        net.load_state_dict(ckpt["state_dict"])
    else:
        epoch_start = 1
        # load imagenet resnet50 backbone
        resnet = torchvision.models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        net_dict = net.state_dict()

        for k in new_state_dict.keys():
            if k in net_dict.keys() and not k.startswith("fc"):
                net_dict[k] = new_state_dict[k]

        net.load_state_dict(net_dict)

    print("NUMBER OF CUDA DEVICES:", torch.cuda.device_count())

    net = net.to(device)
    criterion = yoloLoss().to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # ------------------------------------------
    # AdaBelief Optimizer
    # ------------------------------------------
    optimizer = AdaBelief(
        net.parameters(),
        lr=learning_rate,
        eps=1e-12,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        rectify=False
    )

    # ------------------------------------------
    # Dataset
    # ------------------------------------------
    with open("./Dataset/train.txt") as f:
        train_list = f.readlines()

    train_dataset = Dataset(root, train_list, train=True,
                            transform=[transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=os.cpu_count()
    )

    with open("./Dataset/test.txt") as f:
        test_list = f.readlines()

    test_dataset = Dataset(root, test_list, train=False,
                           transform=[transforms.ToTensor()])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=max(1, batch_size // 2),
        shuffle=False, num_workers=os.cpu_count()
    )

    print(f"NUMBER OF DATA SAMPLES: {len(train_dataset)}")
    print(f"BATCH SIZE: {batch_size}")

    # ------------------------------------------
    # Training Loop
    # ------------------------------------------
    for epoch in range(epoch_start, num_epochs):

        net.train()

        # schedule
        if epoch == 30:
            for g in optimizer.param_groups:
                g["lr"] = 1e-4
        if epoch == 40:
            for g in optimizer.param_groups:
                g["lr"] = 1e-5

        total_loss = 0.0
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.to(device)

            preds = net(images)

            optimizer.zero_grad()
            loss = criterion(preds, targets.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % (
                '%g/%g' % (epoch, num_epochs),
                total_loss / (i + 1), mem
            )
            pbar.set_description(s)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        net.eval()
        val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, targets) in pbar:
                images = images.to(device)
                targets = targets.to(device)

                preds = net(images)
                loss = criterion(preds, targets.float())

                val_loss += loss.item()

        val_loss /= max(1, len(test_loader))
        print(f"Validation Loss: {val_loss:.4f}")

        # save every 10 epochs
        if epoch % 10 == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({"state_dict": net.state_dict()},
                       f"{args.save_dir}/yolov1_{epoch:04d}.pth")

    # final save
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({"state_dict": net.state_dict()},
               f"{args.save_dir}/yolov1_final.pth")


# ------------------------------------------
# ArgumentParser
# ------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--pre_weights", type=str)
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)

    args = parser.parse_args()
    main(args)
