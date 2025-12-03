import os
import cv2
import random
import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, root, file_names, train=True, transform=None, mosaic_prob=0.5):
        self.root = root
        self.train = train
        self.transform = transform
        self.mosaic_prob = mosaic_prob

        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')

        self.names = [x.strip() for x in file_names]

    # --------------------------
    # Load image + label
    # --------------------------
    def load_image_label(self, name):
        img_path = f"{self.root_images}/{name}.jpg"
        lbl_path = f"{self.root_labels}/{name}.txt"

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        boxes = []
        labels = []

        with open(lbl_path) as f:
            objects = f.readlines()

        for obj in objects:
            c, x1, y1, x2, y2 = map(float, obj.split())
            boxes.append([x1, y1, x2, y2])
            labels.append(int(c))

        return img, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    # --------------------------
    # Clip & sanitize boxes
    # --------------------------
    def sanitize_boxes(self, boxes, labels, w, h, min_size=2):
        if len(boxes) == 0:
            return boxes, labels

        # clip
        boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
        boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
        boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
        boxes[:, 3] = boxes[:, 3].clip(0, h - 1)

        # valid box filter
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        keep = (x2 > x1) & (y2 > y1) & ((x2 - x1) >= min_size) & ((y2 - y1) >= min_size)

        if keep.sum() == 0:
            return boxes[:0], labels[:0]

        return boxes[keep], labels[keep]

    # --------------------------
    # Mosaic Augmentation
    # --------------------------
    def mosaic_augmentation(self, idx):
        indices = [idx] + random.sample(range(len(self.names)), 3)

        s = self.image_size
        yc = random.randint(int(0.25 * s), int(0.75 * s))
        xc = random.randint(int(0.25 * s), int(0.75 * s))

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []

        for i, index in enumerate(indices):
            name = self.names[index]
            img, boxes, labels = self.load_image_label(name)

            h, w = img.shape[:2]

            # random scale
            scale = random.uniform(0.5, 1.5)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

            if i == 0:
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x2a, y2a = xc, yc
                x1b, y1b = max(w - (xc - x1a), 0), max(h - (yc - y1a), 0)
            elif i == 1:
                x1a, y1a = xc, max(yc - h, 0)
                x2a, y2a = min(xc + w, s), yc
                x1b, y1b = 0, max(h - (yc - y1a), 0)
            elif i == 2:
                x1a, y1a = max(xc - w, 0), yc
                x2a, y2a = xc, min(yc + h, s)
                x1b, y1b = max(w - (xc - x1a), 0), 0
            else:
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + w, s), min(yc + h, s)
                x1b, y1b = 0, 0

            # paste image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a),
                                              x1b:x1b + (x2a - x1a)]

            # adjust boxes
            pad_x = x1a - x1b
            pad_y = y1a - y1b

            for bi, box in enumerate(boxes):
                x1 = box[0] + pad_x
                y1 = box[1] + pad_y
                x2 = box[2] + pad_x
                y2 = box[3] + pad_y

                mosaic_boxes.append([x1, y1, x2, y2])
                mosaic_labels.append(labels[bi])

        mosaic_boxes = np.array(mosaic_boxes, dtype=np.float32)
        mosaic_labels = np.array(mosaic_labels, dtype=np.int64)

        # sanitize
        mosaic_boxes, mosaic_labels = self.sanitize_boxes(
            mosaic_boxes, mosaic_labels, s, s
        )

        return mosaic_img, mosaic_boxes, mosaic_labels

    # --------------------------
    # YOLOv1 encoder (SAFE VERSION)
    # --------------------------
    def encode(self, boxes, labels):
        S = 14
        target = torch.zeros((S, S, 30), dtype=torch.float32)

        if len(boxes) == 0:
            return target

        # (1) normalize (0~1)
        boxes = boxes.copy()
        boxes[:, [0, 2]] /= self.image_size
        boxes[:, [1, 3]] /= self.image_size

        boxes = boxes.clip(0.0, 1.0 - 1e-6)

        # (2) cxcywh
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        cell = 1.0 / S

        for i in range(len(boxes)):
            cx, cy = cxcy[i]

            grid_x = int(np.clip(cx / cell, 0, S - 1))
            grid_y = int(np.clip(cy / cell, 0, S - 1))

            delta_x = (cx - grid_x * cell) / cell
            delta_y = (cy - grid_y * cell) / cell

            target[grid_y, grid_x, 4] = 1.0
            target[grid_y, grid_x, 9] = 1.0

            # bbox1, bbox2
            target[grid_y, grid_x, :2] = torch.tensor([delta_x, delta_y])
            target[grid_y, grid_x, 2:4] = torch.tensor(wh[i])
            target[grid_y, grid_x, 5:7] = torch.tensor([delta_x, delta_y])
            target[grid_y, grid_x, 7:9] = torch.tensor(wh[i])

            target[grid_y, grid_x, 10 + labels[i]] = 1.0

        return target

    # --------------------------
    # __getitem__
    # --------------------------
    def __getitem__(self, idx):
        if self.train and random.random() < self.mosaic_prob:
            img, boxes, labels = self.mosaic_augmentation(idx)
        else:
            name = self.names[idx]
            img, boxes, labels = self.load_image_label(name)

        # resize to 448×448
        img = cv2.resize(img, (self.image_size, self.image_size))
        h, w = img.shape[:2]

        # sanitize AFTER resize
        boxes, labels = self.sanitize_boxes(boxes, labels, w, h)

        # encode
        target = self.encode(boxes, labels)

        # to RGB
        img = img[:, :, ::-1].copy() / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        return img, target

    def __len__(self):
        return len(self.names)
