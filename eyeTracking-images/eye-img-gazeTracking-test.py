import cv2
from gaze_tracking import GazeTracking

# GazeTracking 초기화
gaze = GazeTracking()

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

# 이미지 좌우 결합
display = cv2.hconcat([img1, img2])

# 웹캠 켜기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 반전 및 분석
    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)

    # 시선 판단
    text = ""
    if gaze.is_right():
        text = "👉 오른쪽 그림을 보고 있어요"
    elif gaze.is_left():
        text = "👈 왼쪽 그림을 보고 있어요"
    elif gaze.is_center():
        text = "🔍 정중앙을 보고 있어요"
    elif gaze.is_blinking():
        text = "😴 눈을 감고 있어요"

    # 텍스트 이미지 위에 표시
    result = display.copy()
    cv2.putText(result, text, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

    # 시선 표시용 웹캠도 함께 보여줌
    webcam_view = gaze.annotated_frame()
    webcam_view = cv2.resize(webcam_view, (400, 300))
    result[10:310, 20:420] = webcam_view

    # 결과 출력
    cv2.imshow("Art Eye Tracker", result)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
