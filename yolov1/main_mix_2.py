
# modern_yolov1_fixed.py
# Fully working Modern YOLOv1 with ResNet50, SPP, Fused Neck, 14x14 output

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

IMG_SIZE = 448
GRID = 14
CLASSES = 20

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2,0,1)

class YOLOv1Dataset(Dataset):
    def __init__(self, root, names):
        self.root = root
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(f"{self.root}/Images/{name}.jpg")
        if img is None:
            raise FileNotFoundError(name)
        target = self.load_label(f"{self.root}/Labels/{name}.txt")
        return preprocess(img), target

    def load_label(self, path):
        S = GRID
        tgt = torch.zeros((S,S,30))
        with open(path) as f:
            for line in f:
                c,x1,y1,x2,y2 = map(float,line.split())
                x1/=IMG_SIZE; x2/=IMG_SIZE; y1/=IMG_SIZE; y2/=IMG_SIZE
                x1 = min(max(x1,0),0.9999)
                y1 = min(max(y1,0),0.9999)
                x2 = min(max(x2,0),0.9999)
                y2 = min(max(y2,0),0.9999)
                w = max(x2-x1,1e-6)
                h = max(y2-y1,1e-6)
                cx = (x1+x2)/2
                cy = (y1+y2)/2
                cx = min(max(cx,0),0.9999)
                cy = min(max(cy,0),0.9999)
                i=int(cx*S); j=int(cy*S)
                if i<0 or i>=S or j<0 or j>=S: continue
                tgt[j,i,4]=1; tgt[j,i,9]=1
                tgt[j,i,10+int(c)] = 1
                offx = cx*S - i; offy = cy*S - j
                tgt[j,i,0:2]=torch.tensor([offx,offy])
                tgt[j,i,2:4]=torch.tensor([w,h])
                tgt[j,i,5:7]=tgt[j,i,0:2]
                tgt[j,i,7:9]=tgt[j,i,2:4]
        return tgt

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        self.stage1 = nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool)
        self.c2 = r.layer1   # 256 ch, 112x112
        self.c3 = r.layer2   # 512 ch, 56x56
        self.c4 = r.layer3   # 1024 ch, 28x28
        self.c5 = r.layer4   # 2048 ch, 14x14

    def forward(self,x):
        x=self.stage1(x)
        c2=self.c2(x)
        c3=self.c3(c2)
        c4=self.c4(c3)
        c5=self.c5(c4)
        return c4,c5  # only c4+c5 used

class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1=nn.MaxPool2d(5,1,2)
        self.p2=nn.MaxPool2d(9,1,4)
        self.p3=nn.MaxPool2d(13,1,6)
    def forward(self,x):
        return torch.cat([x,self.p1(x),self.p2(x),self.p3(x)],1)

class FuseNeck(nn.Module):
    def __init__(self, in_c4=1024, in_spp=2048*4, out=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c4+in_spp,out,1),
            nn.BatchNorm2d(out),
            nn.ReLU(True)
        )
    def forward(self,c4,spp):
        c4_up = F.interpolate(c4, size=spp.shape[2:], mode="nearest")
        return self.conv(torch.cat([c4_up,spp],1))

class DecoupledHead(nn.Module):
    def __init__(self, ch=512, B=2, C=20):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1), nn.ReLU(True),
            nn.Conv2d(ch,B*5,1)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1), nn.ReLU(True),
            nn.Conv2d(ch,C,1)
        )
    def forward(self,x):
        r=self.reg(x)
        c=self.cls(x)
        o=torch.cat([r,c],1)  # [B,30,H,W]
        return o.permute(0,2,3,1)

class CIOU(nn.Module):
    def forward(self,p,t):
        px,py,pw,ph = p.T
        tx,ty,tw,th = t.T
        px1,py1=px-pw/2,py-ph/2
        px2,py2=px+pw/2,py+ph/2
        tx1,ty1=tx-tw/2,ty-th/2
        tx2,ty2=tx+tw/2,ty+th/2
        ix1,iy1 = torch.max(px1,tx1),torch.max(py1,ty1)
        ix2,iy2 = torch.min(px2,tx2),torch.min(py2,ty2)
        inter=(ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
        up=pw*ph; ut=tw*th
        union=up+ut-inter+1e-6
        iou=inter/union
        rho=(px-tx)**2+(py-ty)**2
        c=(torch.max(px2,tx2)-torch.min(px1,tx1))**2 + (torch.max(py2,ty2)-torch.min(py1,ty1))**2 +1e-6
        v=(4/np.pi**2)*(torch.atan(tw/th)-torch.atan(pw/ph))**2
        a=v/(1-iou+v+1e-6)
        ciou=iou - rho/c - a*v
        return (1-ciou).mean()

class FocalConf(nn.Module):
    def __init__(self, g=2,a=0.25):
        super().__init__()
        self.g=g; self.a=a
    def forward(self,p,t):
        b=F.binary_cross_entropy(p,t,reduction='none')
        pt=torch.exp(-b)
        return (self.a*((1-pt)**self.g)*b).mean()

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ciou=CIOU()
        self.focal=FocalConf()
        self.cls=nn.BCEWithLogitsLoss()
    def forward(self,pred,tgt):
        obj = tgt[...,4]>0
        box_p=pred[...,:4][obj]
        box_t=tgt[...,:4][obj]
        conf_p=torch.sigmoid(pred[...,4])
        conf_t=tgt[...,4]
        cls_p=pred[...,10:]
        cls_t=tgt[...,10:]
        Lb=self.ciou(box_p,box_t) if box_p.numel()>0 else torch.tensor(0.,device=pred.device)
        Lc=self.focal(conf_p,conf_t)
        Lcls=self.cls(cls_p[obj],cls_t[obj]) if obj.any() else torch.tensor(0.,device=pred.device)
        return Lb+Lc+Lcls

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=ResNet50Backbone()
        self.spp=SPP()
        self.neck=FuseNeck()
        self.head=DecoupledHead()
    def forward(self,x):
        c4,c5 = self.backbone(x)
        s=self.spp(c5)
        f=self.neck(c4,s)
        return self.head(f)  # [B,14,14,30]

@torch.no_grad()
def eval_loss(model,criterion,loader,device):
    model.eval()
    tot=0;n=0
    for img,tgt in loader:
        img,tgt=img.to(device),tgt.to(device)
        p=model(img)
        L=criterion(p,tgt)
        tot+=L.item(); n+=1
    return tot/max(n,1)
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())
    print("DEVICE:", device)

    # Load file lists
    with open('./Dataset/train.txt') as f:
        train_names = [x.strip() for x in f]
    with open('./Dataset/test.txt') as f:
        test_names = [x.strip() for x in f]

    # Dataset
    train_dataset = YOLOv1Dataset('./Dataset', train_names)
    test_dataset  = YOLOv1Dataset('./Dataset', test_names)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=os.cpu_count())
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False,
                              num_workers=os.cpu_count())

    # Model
    model = Model().to(device)
    criterion = YoloLoss().to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer (same pattern as original main.py)
    learning_rate = 0.001
    params = []
    for key, value in dict(model.named_parameters()).items():
        if key.startswith('backbone'):
            params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Training Loop
    num_epochs = 50
    best_val = 1e9

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {8}')

    for epoch in range(1, num_epochs+1):

        # LR schedule identical to original
        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

        model.train()
        total_loss = 0.
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))

        progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress:
            images = images.to(device)
            target = target.to(device)

            pred = model(images)
            loss = criterion(pred, target.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 
                             if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs),
                                               total_loss / (i + 1), mem)
            progress.set_description(s)

        # Validation
        model.eval()
        validation_loss = 0.
        with torch.no_grad():
            val_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for j, (images, target) in val_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = model(images)
                loss = criterion(prediction, target.float())
                validation_loss += loss.item()

        validation_loss /= len(test_loader)
        print(f'Validation_Loss:{validation_loss:07.3}')

        # Save best based on validation loss
        if validation_loss < best_val:
            best_val = validation_loss
            torch.save({'state_dict': model.state_dict()},
                       f'./weights/yolov1_mix_2_best.pth')
            print(f"Best Model Updated at Epoch {epoch}, Val Loss = {validation_loss:.4f}")

        # Save every 10 epochs like original
        if (epoch % 10) == 0:
            torch.save({'state_dict': model.state_dict()},
                       f'./weights/yolov1_mix_2_{epoch:04d}.pth')

    # Final save
    torch.save({'state_dict': model.state_dict()}, './weights/yolov1_mix_2_final.pth')

if __name__=="__main__":
    train_model()
