from pathlib import Path  # 꼭 추가!!
from ultralytics import YOLO
from PIL import Image

# (1) YOLOv8 모델 로드
detector = YOLO('yolov8n.pt')  # 'n'은 가장 빠른 nano 모델

# (2) 객체 탐지하고 crop하는 함수
def detect_and_crop_objects(image_path: str, save_dir: str):
    image = Image.open(image_path).convert("RGB")
    results = detector.predict(source=image_path, save=False, conf=0.3)[0]  # conf=0.3은 신뢰도 기준

    crops = []
    
    for idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):  # 감지된 박스 순회
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        
        crop_filename = f"{Path(image_path).stem}_crop{idx}.jpg"
        crop_save_path = Path(save_dir) / crop_filename
        
        crop.save(crop_save_path)  # 크롭된 조각 저장
        
        crops.append({
            "crop_path": str(crop_save_path),
            "bbox": [x1, y1, x2, y2]
        })
    
    return crops

# (3) 폴더 설정
image_dir = Path("../test-images/")        # 원본 이미지 폴더
crop_save_dir = Path("./cropped_images/")  # 크롭된 이미지 저장 폴더
crop_save_dir.mkdir(exist_ok=True)

# (4) 이미지 하나하나 돌면서 크롭하기
for img_file in image_dir.glob("*.[jp][pn]g"):  # jpg, jpeg, png
    crops = detect_and_crop_objects(str(img_file), str(crop_save_dir))
    print(f"{img_file.name} 에서 {len(crops)}개 객체를 크롭했습니다.")
