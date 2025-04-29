#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:12:12 2025

@author: kangchaewon
"""

import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

# 모델 로드
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 이미지 임베딩 함수
def image_embedding(img_path: Path) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)  # [1, 512]
    emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 정규화
    return emb.cpu().numpy().astype("float32")

# 이미지 디렉토리 경로
# image_dir = Path("../test-images/")

# Faiss 인덱스 불러오기
index = faiss.read_index("image_clip.index")

# 쿼리 이미지의 임베딩을 구하는 함수
def search_similar_image(query_img_path: Path, index: faiss.Index, top_k=1):
    query_embedding = image_embedding(query_img_path)  # 쿼리 이미지 임베딩
    query_embedding = query_embedding.astype('float32')  # 타입 변환

    # Faiss 인덱스를 사용하여 가장 유사한 이미지 검색
    distances, indices = index.search(query_embedding, top_k)  # top_k=1로 설정하여 가장 유사한 하나만 찾음

    return distances, indices

# 검색할 이미지
query_img_path = Path("../test-images/test-1.jpg")  # 검색할 이미지 경로를 지정

# 검색
distances, indices = search_similar_image(query_img_path, index)

# 결과 출력
print(f"Query image: {query_img_path.name}")
print(f"Most similar image index: {indices[0][0]}")  # 가장 유사한 이미지의 인덱스
print(f"Distance (similarity score): {distances[0][0]}")  # 유사도 점수
