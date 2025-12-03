import cv2
import numpy as np
import torch
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, filelist, train=True, mosaic_prob=0.5, mixup_prob=0.3):
        self.root = root
        self.filelist = [x.strip() for x in filelist]
        self.train = train
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.image_size = 448
        self.S = 14

    def __len__(self):
        return len(self.filelist)

    # -------------------------------------------------------------
    # load image + label
    # -------------------------------------------------------------
    def load_item(self, index):
        name = self.filelist[index]
        img_path = f"{self.root}/Images/{name}.jpg"
        label_path = f"{self.root}/Labels/{name}.txt"

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f.readlines():
                c, x1, y1, x2, y2 = map(int, line.split())
                boxes.append([x1, y1, x2, y2])
                labels.append(c)

        return img, np.array(boxes), np.array(labels)

    # -------------------------------------------------------------
    # Mixup augmentation
    # -------------------------------------------------------------
    def mixup(self, img1, boxes1, labels1, img2, boxes2, labels2, alpha=0.5):
        lam = np.random.beta(alpha, alpha)

        img = (img1 * lam + img2 * (1 - lam)).astype(np.uint8)
        boxes = np.concatenate([boxes1, boxes2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)

        return img, boxes, labels

    # -------------------------------------------------------------
    # Mosaic augmentation
    # -------------------------------------------------------------
    def mosaic(self):
        indices = [random.randint(0, len(self.filelist)-1) for _ in range(4)]

        final_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        final_boxes = []
        final_labels = []

        half = self.image_size // 2

        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_item(idx)

            img = cv2.resize(img, (half, half))

            row = (i // 2) * half
            col = (i % 2) * half
            final_img[row:row+half, col:col+half] = img

            boxes = boxes * 0.5
            for b in boxes:
                b[0] += col
                b[2] += col
                b[1] += row
                b[3] += row

            final_boxes.append(boxes)
            final_labels.append(labels)

        final_boxes = np.concatenate(final_boxes, axis=0)
        final_labels = np.concatenate(final_labels, axis=0)

        return final_img, final_boxes, final_labels

    # -------------------------------------------------------------
    # Encoding YOLO target
    # -------------------------------------------------------------
    def encode(self, boxes, labels):
        S = self.S
        target = np.zeros((S, S, 30), dtype=np.float32)

        if len(boxes) == 0:
            return torch.tensor(target)

        boxes = boxes.astype(np.float32)
        boxes[:, 0] /= self.image_size
        boxes[:, 1] /= self.image_size
        boxes[:, 2] /= self.image_size
        boxes[:, 3] /= self.image_size

        boxes = boxes.clip(0, 1)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            grid_x = int(cx * S)
            grid_y = int(cy * S)

            grid_x = min(grid_x, S-1)
            grid_y = min(grid_y, S-1)

            target[grid_y, grid_x, 0:2] = [cx*S - grid_x, cy*S - grid_y]
            target[grid_y, grid_x, 2:4] = [w, h]
            target[grid_y, grid_x, 4] = 1

            target[grid_y, grid_x, 5:7] = [cx*S - grid_x, cy*S - grid_y]
            target[grid_y, grid_x, 7:9] = [w, h]
            target[grid_y, grid_x, 9] = 1

            target[grid_y, grid_x, 10 + labels[i]] = 1

        return torch.tensor(target)

    # -------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------
    def __getitem__(self, idx):
        if self.train and random.random() < self.mosaic_prob:
            img, boxes, labels = self.mosaic()
        else:
            img, boxes, labels = self.load_item(idx)

        # Mixup
        if self.train and random.random() < self.mixup_prob:
            idx2 = random.randint(0, len(self.filelist)-1)
            img2, boxes2, labels2 = self.load_item(idx2)
            img, boxes, labels = self.mixup(img, boxes, labels, img2, boxes2, labels2)

        img = img[:, :, ::-1]  # BGR → RGB
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.tensor(img, dtype=torch.float32)

        target = self.encode(boxes, labels)

        return img, target
