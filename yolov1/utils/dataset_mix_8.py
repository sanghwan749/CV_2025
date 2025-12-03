import cv2
import numpy as np

class DatasetMix8:
    def __init__(self, img_dir, name_list, S=14, train=True):
        self.img_dir = img_dir
        self.name_list = name_list
        self.S = S
        self.train = train

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx].strip()
        img_path = f"{self.img_dir}/Images/{name}.jpg"
        label_path = f"{self.img_dir}/Labels/{name}.txt"

        img = cv2.imread(img_path)
        img = cv2.resize(img, (448, 448))
        img = img[:, :, ::-1]  # BGR → RGB

        objs = []
        with open(label_path) as f:
            for line in f.readlines():
                c, x1, y1, x2, y2 = map(int, line.split())
                objs.append((c, x1, y1, x2, y2))

        target = self.encode(objs)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        return img, target


    def encode(self, objs):
        S = self.S
        target = np.zeros((S, S, 30), dtype=np.float32)

        for c, x1, y1, x2, y2 in objs:

            cx = (x1 + x2) / 2 / 448
            cy = (y1 + y2) / 2 / 448
            w  = (x2 - x1) / 448
            h  = (y2 - y1) / 448

            sw = np.sqrt(w)
            sh = np.sqrt(h)

            gx = min(int(cx * S), S - 1)
            gy = min(int(cy * S), S - 1)

            # cell offset normalize
            tx = cx * S - gx
            ty = cy * S - gy

            # ---- box1 ----
            target[gy, gx, 0] = tx
            target[gy, gx, 1] = ty
            target[gy, gx, 2] = sw
            target[gy, gx, 3] = sh
            target[gy, gx, 4] = 1.0  # objectness for box1

            # ---- box2 ---- (GT 등록 필수)
            target[gy, gx, 5] = tx
            target[gy, gx, 6] = ty
            target[gy, gx, 7] = sw
            target[gy, gx, 8] = sh
            target[gy, gx, 9] = 1.0  # objectness for box2 (for IoU choice)

            # ---- class ----
            target[gy, gx, 10 + c] = 1.0

        return target
