import cv2
from ultralytics import YOLO
from gaze_tracking import GazeTracking

# YOLO Segmentation 모델 로드
model = YOLO("yolov8n-seg.pt")  # yolov8s-seg.pt, yolov8m-seg.pt 등으로 변경 가능

# 시선 추적기 초기화
gaze = GazeTracking()

# 이미지 로드 및 리사이즈
image_path = "./test-images/image-1.jpeg"
img = cv2.imread(image_path)
screen_width, screen_height = 1280, 720
img = cv2.resize(img, (screen_width, screen_height))

# YOLO 세그멘테이션 수행
results = model(img, conf=0.3)[0]  # 첫 번째 결과만 사용

# 클래스 및 마스크 정보 추출
class_names = model.names
masks = results.masks.data.cpu()  # (N, H, W)
boxes = results.boxes.xyxy.cpu().numpy()
classes = results.boxes.cls.cpu().numpy().astype(int)

# 시각화용 결과 이미지 생성
annotated_img = results.plot()

# 객체 마스크와 클래스명 저장
object_masks = []
for i in range(len(masks)):
    mask = masks[i]
    class_id = classes[i]
    label = class_names[class_id]
    object_masks.append((mask, label))

# 웹캠으로 시선 추적
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 셀카처럼 반전
    gaze.refresh(frame)

    gaze_x = gaze.horizontal_ratio()
    gaze_y = gaze.vertical_ratio()

    result = annotated_img.copy()  # 원본 이미지 복사
    text = ""

    if gaze_x is not None and gaze_y is not None:
        # 시선 위치 (이미지 기준 좌표)
        screen_x = int(gaze_x * screen_width)
        screen_y = int(gaze_y * screen_height)

        # 표시용 동그라미
        cv2.circle(result, (screen_x, screen_y), 8, (0, 0, 255), -1)

        found_obj = None
        for mask, label in object_masks:
            h_mask, w_mask = mask.shape

            # 마스크 좌표 기준 시선 위치 계산
            mask_x = int(gaze_x * w_mask)
            mask_y = int(gaze_y * h_mask)

            # 유효 좌표인지 확인 후 검사
            if 0 <= mask_x < w_mask and 0 <= mask_y < h_mask:
                if mask[mask_y, mask_x] > 0.5:
                    found_obj = label
                    break

        if found_obj:
            text = f"👁️ 바라보는 객체: {found_obj}"
        else:
            text = "👁️ 객체 영역이 아님"
    else:
        text = "😴 눈 감았거나 인식 불가"

    # 결과 출력
    cv2.putText(result, text, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Segmentation + Gaze Tracking", result)

    if cv2.waitKey(1) == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
