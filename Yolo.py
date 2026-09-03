import cv2
from ultralytics import YOLOE
model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(["egg", "stone", "sponge"])
cap = cv2.VideoCapture(0)
success, img = cap.read()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
results = model.predict(imgGray, imgsz=320, conf=0.2, device="cpu")
print(results)
