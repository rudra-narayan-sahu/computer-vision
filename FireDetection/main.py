import cv2
from ultralytics import YOLO
img=cv2.imread("fireIMg.jpg")

new_model=YOLO('yolov8n_fire_smoke.pt')
res=new_model(img)
res[0].show()