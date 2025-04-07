#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:22:43 2025

@author: kangchaewon
"""

from ultralytics import YOLO
import cv2

# 모델 로드 (YOLOv8n은 가장 가벼운 모델)
model = YOLO("yolov8n.pt")  # yolov8s.pt / yolov8m.pt 등으로 변경 가능

# 이미지 경로
image_path = "../test-images/image-1.jpeg"

# 이미지 감지 & 민감도 낮추기
# 예: confidence threshold를 0.1로 낮추기
results = model(image_path, conf=0.1)


# 결과 이미지 저장 및 시각화
for result in results:
    # 결과 이미지 저장 (선택)
    result.save(filename="result.jpg")

    # OpenCV로 이미지 출력
    annotated_frame = result.plot()  # 바운딩 박스 그려진 이미지
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
