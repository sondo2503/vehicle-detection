import queue
import torch
import numpy as np
import cv2
import os
from .utils.general import non_max_suppression, scale_boxes
from .models.experimental import attempt_load

class Detector:
    def __init__(self, weight_path, img_size=640):

        self._img_size = img_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._half = self._device.type != 'cpu'
        self._model = attempt_load(weight_path, device=self._device)
        self._stride = int(self._model.stride.max())
        self._names = self._model.module.names if hasattr(self._model, 'module') else self._model.names
        if torch.cuda.is_available():
            self._model.cuda()
            if self._half:
                self._model.half()

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def strip_optimizer(f='best_last.pt', s=''):  # from utils.general import *; strip_optimizer()
        # Strip optimizer from 'f' to finalize training, optionally save as 's'
        x = torch.load(f, map_location=torch.device('cpu'))
        if x.get('ema'):
            x['model'] = x['ema']  # replace model with ema
        for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
            x[k] = None
        x['epoch'] = -1
        x['model'].half()  # to FP16
        for p in x['model'].parameters():
            p.requires_grad = False
        torch.save(x, s or f)
        mb = os.path.getsize(s or f) / 1E6  # filesize
        print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

    def makeInputModel(self, img_src):
        inputModelIMG = self.letterbox(img_src, self._img_size, auto=True, stride=self._stride)[0]
        img = inputModelIMG[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict_IMG(self, img):
        inputModelIMG = self.makeInputModel(img)
        pred = self._model(inputModelIMG, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.25, classes=None, agnostic=False)
        return pred, inputModelIMG
    
    def detect_vehicle(self, imgSrc, file_name, thresh=0.8):
        pred, img = self.predict_IMG(imgSrc)
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in det:
                    conf = float(conf)
                    cls = int(cls)
                    if conf > thresh:
                        point1_x = point4_x = float(xyxy[0])
                        point1_y = point2_y = float(xyxy[1])
                        point2_x = point3_x = float(xyxy[2])
                        point3_y = point4_y = float(xyxy[3])
                        results.append([file_name, cls, conf, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])
                    else:
                        for i in range(9):
                            if results:
                                break
                            else:
                                thresh -= 0.1
                                if conf > thresh:
                                    point1_x = point4_x = float(xyxy[0])
                                    point1_y = point2_y = float(xyxy[1])
                                    point2_x = point3_x = float(xyxy[2])
                                    point3_y = point4_y = float(xyxy[3])
                                    results.append([file_name, cls, conf, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])
        return results

                    