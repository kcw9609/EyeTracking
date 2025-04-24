import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 세그멘테이션 모델
model = YOLO("yolov8n-seg.pt")

# 이미지 불러오기
image_path = "../test-images/image-3.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
screen_width, screen_height = 1280, 720
image = cv2.resize(image, (screen_width, screen_height))
# 모델 추론
results = model(image, conf=0.3)[0]
class_names = model.names

# 마스크 및 클래스
masks = results.masks.data.cpu().numpy()  # (N, H, W)
classes = results.boxes.cls.cpu().numpy().astype(int)

# 무작위 색상 설정
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

# 결과 이미지 복사본
seg_image = image.copy()
object_regions = []

# 각 객체 마스크 적용
for i, mask in enumerate(masks):
    class_id = classes[i]
    color = colors[class_id]
    binary_mask = (mask > 0.5).astype(np.uint8)

    # 마스크를 원본 이미지 크기에 맞게 resize
    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    # 색상 입히기
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = binary_mask * color[c]

    seg_image = cv2.addWeighted(seg_image, 1.0, colored_mask, 0.5, 0)
    object_regions.append((binary_mask, class_names[class_id]))

# 마우스 클릭 콜백
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for mask, label in object_regions:
            if mask[y, x] == 1:
                print(f"🖱️ ({x}, {y}) → '{label}' 객체 클릭")
                return
        print(f"🖱️ ({x}, {y}) → 객체 없음")

cv2.namedWindow("YOLOv8 Segmentation Click")
cv2.setMouseCallback("YOLOv8 Segmentation Click", on_mouse)

while True:
    cv2.imshow("YOLOv8 Segmentation Click", seg_image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
