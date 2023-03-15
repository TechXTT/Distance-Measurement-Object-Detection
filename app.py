#Detect human face in a video stream using Yolov7 and OpenCV
#After detecting the face, calculate the distance between the camera and the face
# and display the distance on the screen in real time in cm

#Import the necessary packages

import os
import sys
sys.path.append('./yolov7')
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#Define the global variables
classes_to_filter = ['train'] 

opt = {
    "weights" : "weights/yolov7.pt",
    "source" : "yolov7/data/coco.coco.yaml",
    "img_size" : 640,
    "conf_thres" : 0.4,
    "iou_thres" : 0.5,
    "device" : "cpu",
    "classes" : classes_to_filter,
}

weights, imgsz = opt['weights'], opt['img_size']
device = select_device(opt['device'])
half = device.type != 'cpu'
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))



#Define the function to calculate the distance between the camera and the face
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    focalLength = (1280 * 28) / knownWidth
    distance = (knownWidth * focalLength) / perWidth
    print(f"Distance: {distance} cm, Focal Length: {focalLength} px")
    return distance
    
#Define the function for the bounding box
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

#Define the class for the camera
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def release(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        return frame

#Define the function to detect the face and calculate the distance
def detect_face_distance(frame):
    img = letterbox(frame, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=[0], agnostic=False)
    t2 = time_synchronized()
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # Calculate the distance between the camera and the face

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):
                distance = distance_to_camera(knownWidth=20, focalLength=0, perWidth=xyxy[2]-xyxy[0])
                label = f'{names[int(cls)]} {conf:.2f}'
                label = f'{label} {distance} cm'
                # print(label)
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

    return frame

#Define the function to show the video
def gen(camera):
    # use opencv to display the video
    while True:
        frame = camera.get_frame()
        frame = detect_face_distance(frame)
        cv2.imshow('frame', frame)
        if 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    videoCam = VideoCamera()
    gen(videoCam)

