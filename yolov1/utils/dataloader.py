import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ==========================================================
# YOLOv1 Dataset with internal normalization
# ==========================================================

class YOLOv1Dataset(Dataset):
    def __init__(self, img_dir, label_dir, list_file,
                 S=14, B=2, C=20, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        # 이미지 리스트 로드
        with open(list_file, 'r') as f:
            self.image_list = f.read().strip().split()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_id = self.image_list[idx]

        # 이미지 파일 로드
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        img = Image.open(img_path).convert("RGB")

        W, H = img.size  # 이미지 원본 크기

        # =============================================
        #  라벨 읽기 (class xmin ymin xmax ymax)
        # =============================================
        label_path = os.path.join(self.label_dir, img_id + ".txt")
        boxes = []

        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = line.strip().split()
                    cls = int(values[0])
                    xmin, ymin, xmax, ymax = map(float, values[1:])

                    # --- Normalize ---
                    x_center = ((xmin + xmax) / 2) / W
                    y_center = ((ymin + ymax) / 2) / H
                    w = (xmax - xmin) / W
                    h = (ymax - ymin) / H

                    boxes.append([cls, x_center, y_center, w, h])

        # =============================================
        #  이미지 변환(Resize or augmentation)
        # =============================================
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0

        # =============================================
        #  YOLOv1 타깃 텐서 생성
        #  shape = (S, S, 5*B + C)
        # =============================================
        target = torch.zeros((self.S, self.S, 5 * self.B + self.C))

        for box in boxes:
            cls, xc, yc, w, h = box

            # 박스가 속한 셀 결정
            cell_x = int(xc * self.S)
            cell_y = int(yc * self.S)

            # 셀 내부 좌표
            x_in_cell = xc * self.S - cell_x
            y_in_cell = yc * self.S - cell_y

            # Bounding box 1 (YOLOv1 기본 구조에서는 보통 첫 번째 박스만 이용)
            target[cell_y, cell_x, 0] = x_in_cell
            target[cell_y, cell_x, 1] = y_in_cell
            target[cell_y, cell_x, 2] = w
            target[cell_y, cell_x, 3] = h
            target[cell_y, cell_x, 4] = 1  # confidence

            # 클래스 one-hot
            target[cell_y, cell_x, 5 + cls] = 1

            # 두 번째 박스(B=2)까지 넣고 싶다면 아래 주석 해제
            # offset = 5
            # target[cell_y, cell_x, offset+0] = x_in_cell
            # target[cell_y, cell_x, offset+1] = y_in_cell
            # target[cell_y, cell_x, offset+2] = w
            # target[cell_y, cell_x, offset+3] = h
            # target[cell_y, cell_x, offset+4] = 1  

        return img, target


# ==========================================================
# 사용 예시
# ==========================================================
if __name__ == "__main__":
    dataset = YOLOv1Dataset(
        img_dir="./Dataset/Images",
        label_dir="./Dataset/Labels",
        list_file="./Dataset/train.txt",
        S=7, B=2, C=20
    )

    img, target = dataset[0]
    print("Image shape:", img.shape)
    print("Target shape:", target.shape)
