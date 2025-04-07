import cv2
from gaze_tracking import GazeTracking

# GazeTracking 초기화
gaze = GazeTracking()

# 이미지 불러오기
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
display_base = cv2.hconcat([img1, img2])

# 웹캠 켜기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)
    
    # 화면 복사 (그림 위에 시선 그릴 용도)
    display = display_base.copy()

    # 눈동자 중심 좌표 가져오기
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # 두 눈 평균 좌표 계산
    if left_pupil and right_pupil:
        avg_x = int((left_pupil[0] + right_pupil[0]) / 2)
        avg_y = int((left_pupil[1] + right_pupil[1]) / 2)

        # 웹캠 해상도와 display 해상도 맞추기 위해 스케일링
        webcam_width = frame.shape[1]
        webcam_height = frame.shape[0]
        scale_x = screen_width / webcam_width
        scale_y = screen_height / webcam_height

        scaled_x = int(avg_x * scale_x)
        scaled_y = int(avg_y * scale_y)

        # 이미지에 시선 위치 그리기
        cv2.circle(display, (scaled_x, scaled_y), 20, (0, 255, 0), -1)

        # 어떤 그림 위에 있는지 판단
        if scaled_x < screen_width // 2:
            msg = "👁 왼쪽 그림을 보고 있어요"
        else:
            msg = "👁 오른쪽 그림을 보고 있어요"

        cv2.putText(display, msg, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 3)
    
    # 결과 출력
    cv2.imshow("Gaze Tracker Art", display)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
