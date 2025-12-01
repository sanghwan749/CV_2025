import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.focal_alpha_diou_loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size

    # AdamW 권장 lr
    base_lr = args.lr if args.lr else 1e-4
    learning_rate = base_lr

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet50()

    # pretrained
    if(args.pre_weights != None):
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern,f_name).group(1)
        epoch_start = int(epoch_str) + 1

        net.load_state_dict(torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        
        resnet = torchvision.models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
    
        net_dict = net.state_dict()
        for k in new_state_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = new_state_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    # ============================================================
    # AdamW 파라미터 그룹 구성 (features 레이어는 lr ×3)
    # ============================================================
    params = []
    params_dict = dict(net.named_parameters())

    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': base_lr * 3}]
        else:
            params += [{'params': [value], 'lr': base_lr}]

    # ============================================================
    # AdamW Optimizer
    # ============================================================
    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # ============================================================
    # Dataset Loader
    # ============================================================
    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size // 2,
                                              shuffle=False,
                                              num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(epoch_start, num_epochs):
        net.train()

        # -------------------------
        # Learning Rate Decay
        # -------------------------
        if epoch == 30:
            learning_rate = 1e-5
        if epoch == 40:
            learning_rate = 1e-6

        # -------------------------
        # LR 업데이트 (features 레이어 ×3 유지)
        # -------------------------
        for param_group in optimizer.param_groups:
            base_param_lr = param_group['lr']

            # features 그룹인지 판단
            if np.isclose(base_param_lr, base_lr * 3, atol=1e-9):
                param_group['lr'] = learning_rate * 3
            else:
                param_group['lr'] = learning_rate

        # -------------------------
        # Training
        # -------------------------
        total_loss = 0.
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))

        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)
            optimizer.zero_grad()

            loss = criterion(pred, target.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs),
                                                total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        # -------------------------
        # Validation
        # -------------------------
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = net(images)
                loss = criterion(prediction, target)
                validation_loss += loss.item()
            
        validation_loss /= len(test_loader)
        print(f'Validation_Loss:{validation_loss:07.3f}')

        if (epoch % 10) == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, f'./weights/yolov1_AdamW_AlphaDIoU_{epoch:04d}.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final_AdamW_AlphaDIoU.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str)
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    
    main(args)
