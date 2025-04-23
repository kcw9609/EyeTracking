from pathlib import Path
from PIL import Image
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
import json

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def image_embedding(img_path: Path) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)   # [1, 512]
    emb = emb / emb.norm(dim=-1, keepdim=True)          # L2 정규화
    return emb.cpu().numpy().astype("float32")

# 이미지 디렉토리 경로
image_dir = Path("../test-images/")

# 설명을 포함한 메타데이터
image_data = {
    "pic1.jpg": "Peers, flowers, and water kettles",
    "monet_waterlilies.png": "A famous painting by Claude Monet",
    "starry_night.jpg": "A painting by Vincent van Gogh"
}

# 임베딩 벡터와 메타데이터 저장할 리스트
vectors, ids, descriptions = [], [], []

# 이미지 파일에 대해 임베딩 생성하고 메타데이터 저장
# 수정 후 이미지 파일 경로 찾기 (jpeg 파일 추가)
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
for img_file in image_files:
    try:
        vectors.append(image_embedding(img_file))
        ids.append(img_file.name)  # 파일명 저장
        descriptions.append(image_data.get(img_file.name, "No description available"))  # 설명 저장
    except Exception as e:
        print(f"Error processing {img_file}: {e}")

# 임베딩된 벡터들 합치기
if len(vectors) > 0:
    matrix = np.vstack(vectors)  # 벡터가 있을 때만 호출
else:
    print("No vectors to stack. Please check your image files and embeddings.")

# 벡터의 차원 정보 확인
if 'matrix' in locals():
    d = matrix.shape[1]

    # Faiss 인덱스 생성 (코사인 유사도용)
    index = faiss.IndexFlatIP(d)
    index.add(matrix)

    # Faiss 인덱스를 파일로 저장
    faiss.write_index(index, "image_clip.index")

    # 설명과 파일명을 JSON 형식으로 저장
    meta_data = {"ids": ids, "descriptions": descriptions}
    with open("image_meta.json", "w") as f:
        json.dump(meta_data, f)

    print(f"Saved {len(ids)} image vectors with descriptions.")
else:
    print("No embeddings were generated.")

