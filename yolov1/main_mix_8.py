# ============================================================
# main_mix_8.py — YOLOv1 mix_8 Training
# ============================================================
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.nn import resnet50
from utils.dataset_mix_8 import DatasetMix8
from utils.loss_mix_8 import YoloLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50().to(device)
    criterion = YoloLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    with open("./Dataset/train.txt") as f:
        train_list = f.readlines()
    train_dataset = DatasetMix8("./Dataset", train_list, train=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    with open("./Dataset/test.txt") as f:
        val_list = f.readlines()
    val_dataset = DatasetMix8("./Dataset", val_list, train=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    print("Training samples:", len(train_dataset))

    for epoch in range(1, 81):
        model.train()
        total_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}/80] Loss: {total_loss/len(train_loader):.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./weights/yolov1_mix_8_{epoch}.pth")

    torch.save(model.state_dict(), "./weights/yolov1_mix_8_final.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()
