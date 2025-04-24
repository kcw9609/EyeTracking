#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:36:47 2025

@author: kangchaewon
"""

import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolov8n.pt")

# 이미지 경로
image_path = "../test-images/image-1.jpg"

# 이미지 감지 (confidence threshold 낮춤)
results = model(image_path, conf=0.1)
result = results[0]  # 첫 번째 결과만 사용
boxes = result.boxes
class_names = model.names

# 바운딩 박스 리스트로 저장 [(x1, y1, x2, y2, class_id)]
detected_objects = []
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    class_id = int(box.cls[0])
    detected_objects.append((x1, y1, x2, y2, class_id))

# 결과 이미지
annotated_frame = result.plot()

# 마우스 클릭 이벤트 콜백 함수
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        found = False
        for x1, y1, x2, y2, class_id in detected_objects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"({x}, {y}) → 객체: '{class_names[class_id]}' 안에 있습니다.")
                found = True
                break
        if not found:
            print(f"({x}, {y}) → 어떤 객체에도 포함되지 않았습니다.")

# 창 생성 및 마우스 콜백 설정
cv2.namedWindow("YOLOv8 Detection")
cv2.setMouseCallback("YOLOv8 Detection", on_mouse)

# 이미지 출력
while True:
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

cv2.destroyAllWindows()
