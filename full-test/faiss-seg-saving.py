from pathlib import Path
from PIL import Image
import numpy as np
import torch
import faiss
import json
import cv2
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import os

# ===========================
# 모델 로드
# ===========================
model_name = "openai/clip-vit-base-patch32"
device = torch.device('cpu')
clip_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

yolo_model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentation 모델

# ===========================
# 경로 설정
# ===========================
image_dir = Path("../test-images/")   # 원본 이미지 폴더
index_save_path = "./faiss/image_clip.index"
meta_save_path = "./faiss/image_meta.json"
os.makedirs("./faiss", exist_ok=True)

# ===========================
# 함수 정의
# ===========================
def segment_and_crop(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return []

    image = cv2.resize(image, (1280, 720))
    results = yolo_model(image, conf=0.3)[0]

    if results.masks is None:
        return []

    masks = results.masks.data.cpu().numpy()
    cropped_images = []

    for i, mask in enumerate(masks):
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)

        masked_image = np.where(binary_mask_3ch == 1, image, 255)
        x_indices, y_indices = np.where(binary_mask == 1)

        if x_indices.size == 0 or y_indices.size == 0:
            continue

        x_min, x_max = np.min(y_indices), np.max(y_indices)
        y_min, y_max = np.min(x_indices), np.max(x_indices)
        cropped_object = masked_image[y_min:y_max, x_min:x_max]

        if cropped_object.size == 0:
            continue

        cropped_images.append((cropped_object, i))  # 이미지와 인덱스 반환

    return cropped_images

def image_embedding(pil_img: Image.Image) -> np.ndarray:
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

# ===========================
# 메인 실행
# ===========================
vectors, ids, descriptions = [], [], []

image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))

for img_file in image_files:
    print(f"🔍 {img_file.name} 처리 중...")
    crops = segment_and_crop(img_file)

    for cropped_object, crop_idx in crops:
        # OpenCV 이미지를 PIL 이미지로 변환
        pil_cropped = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
        try:
            emb = image_embedding(pil_cropped)
            vectors.append(emb)
            crop_id = f"{img_file.stem}_crop{crop_idx}.jpg"  # 예: image-1_crop0.jpg
            ids.append(crop_id)
            descriptions.append(f"Auto-generated crop from {img_file.name}")
        except Exception as e:
            print(f"Error embedding {img_file.name} crop {crop_idx}: {e}")

# ===========================
# FAISS 인덱스 생성 및 저장
# ===========================
if len(vectors) > 0:
    matrix = np.vstack(vectors)
    d = matrix.shape[1]
    index = faiss.IndexFlatIP(d)  # 코사인 유사도용
    index.add(matrix)

    faiss.write_index(index, index_save_path)
    print(f"✅ FAISS 인덱스 저장 완료: {index_save_path}")

    meta_data = {"ids": ids, "descriptions": descriptions}
    with open(meta_save_path, "w") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 메타데이터 저장 완료: {meta_save_path}")
    print(f"총 {len(ids)}개 crop 객체 인덱싱 완료")
else:
    print("❗ 임베딩된 데이터가 없습니다.")
