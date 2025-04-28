#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:57:21 2025

@author: kangchaewon
"""

from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import faiss
import json
from transformers import CLIPProcessor, CLIPModel

# (1) 모델 로딩
detector = YOLO('yolov8n.pt')  # YOLOv8 nano
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# (2) 임베딩 함수
def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

# (3) 객체 탐지 + 크롭 함수
def detect_and_crop_objects(image_path: str, save_dir: str):
    image = Image.open(image_path).convert("RGB")
    results = detector.predict(source=image_path, save=False, conf=0.3)[0]

    crops = []
    for idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        
        crop_filename = f"{Path(image_path).stem}_crop{idx}.jpg"
        crop_save_path = Path(save_dir) / crop_filename
        crop.save(crop_save_path)
        
        crops.append(str(crop_save_path))
    
    return crops

# (4) 메인 파이프라인
image_dir = Path("../test-images/")        # 원본 이미지 폴더
crop_save_dir = Path("./cropped_images/")  # 크롭 저장 폴더
crop_save_dir.mkdir(exist_ok=True)

vectors = []  # faiss용 전체 벡터 모음
meta_data = []  # 메타데이터 json용

for img_file in image_dir.glob("*.[jp][pn]g"):
    print(f"Processing {img_file.name}...")
    image = Image.open(img_file).convert("RGB")
    
    # (4-1) 전체 이미지 임베딩
    full_emb = get_image_embedding(image)
    vectors.append(full_emb)
    
    # (4-2) 설명 (임시)
    full_desc = f"Placeholder description for {img_file.name}"
    
    # (4-3) 객체 탐지 + 크롭 이미지 만들기
    crop_paths = detect_and_crop_objects(str(img_file), str(crop_save_dir))
    
    crop_info = []
    
    for crop_path in crop_paths:
        crop_image = Image.open(crop_path).convert("RGB")
        crop_emb = get_image_embedding(crop_image)
        vectors.append(crop_emb)
        
        crop_desc = f"Placeholder crop description for {Path(crop_path).name}"
        
        crop_info.append({
            "crop_id": Path(crop_path).name,
            "crop_description": crop_desc
        })
    
    # (4-4) 메타데이터 정리
    meta_data.append({
        "full_image_id": img_file.name,
        "full_image_description": full_desc,
        "crops": crop_info
    })

# (5) 벡터 통합해서 faiss 저장
matrix = np.vstack(vectors)
d = matrix.shape[1]

index = faiss.IndexFlatIP(d)
index.add(matrix)

faiss.write_index(index, "image_clip.index")
with open("image_meta.json", "w") as f:
    json.dump(meta_data, f, indent=2)

print(f"총 {len(vectors)}개 벡터 저장 완료.")
