import cv2
from gaze_tracking import GazeTracking
from ultralytics import YOLO

# GazeTracking 및 YOLO 초기화
gaze = GazeTracking()
model = YOLO("yolov8n.pt")

# 이미지 불러오기 (2개)
img1 = cv2.imread('./test-images/image-1.jpeg')
img2 = cv2.imread('./test-images/image-2.jpeg')

if img1 is None or img2 is None:
    print("❌ 이미지 로드 실패")
    exit()

# 화면 사이즈 지정 및 이미지 리사이즈
screen_width, screen_height = 1280, 720
img_w = screen_width // 2
img_h = screen_height
img1 = cv2.resize(img1, (img_w, img_h))
img2 = cv2.resize(img2, (img_w, img_h))

# YOLO로 객체 인식 (민감도 0.5로 조정)
results1 = model(img1, conf=0.2)[0]
results2 = model(img2, conf=0.2)[0]

# 클래스 이름
class_names = model.names

# 바운딩 박스 그리기 & 정보 저장
detected_objects = []

def draw_and_extract(results, offset_x, image):
    objects = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = class_names[class_id]

        # 좌표 보정해서 저장
        objects.append((x1 + offset_x, y1, x2 + offset_x, y2, class_id))

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return objects

# 각 이미지에 대해 바운딩 박스와 객체 저장
objects1 = draw_and_extract(results1, 0, img1)
objects2 = draw_and_extract(results2, img_w, img2)
detected_objects = objects1 + objects2

# 이미지 병합
display = cv2.hconcat([img1, img2])

# 시선이 어떤 객체 위에 있는지 확인
def get_object_at(x, y):
    for x1, y1, x2, y2, class_id in detected_objects:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return class_names[class_id]
    return None

# 웹캠 켜기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)

    gaze_x = gaze.horizontal_ratio()
    gaze_y = gaze.vertical_ratio()

    result = display.copy()
    text = ""

    if gaze_x is not None and gaze_y is not None:
        screen_x = int(gaze_x * screen_width)
        screen_y = int(gaze_y * screen_height)

        # 시선 위치 표시
        cv2.circle(result, (screen_x, screen_y), 8, (0, 0, 255), -1)

        obj = get_object_at(screen_x, screen_y)
        if obj:
            text = f"👁️ 바라보는 객체: {obj}"
        else:
            text = "👁️ 객체 영역이 아님"
    else:
        text = "😴 눈 감았거나 인식 불가"

    # 텍스트 출력
    cv2.putText(result, text, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    # 화면 출력
    cv2.imshow("Art Eye Tracker + YOLO", result)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
