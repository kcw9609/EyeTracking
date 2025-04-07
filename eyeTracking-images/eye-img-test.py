import cv2
import mediapipe as mp

# 그림 2개 로드
images = []
for i in range(2):
    path = f'./test-images/image-{i+1}.jpeg'
    img = cv2.imread(path)
    if img is None:
        print(f'❌ 이미지 로드 실패: {path}')
    else:
        print(f'✅ 이미지 로드 성공: {path}')
        images.append(img)

print(f'\n총 {len(images)}개 이미지 로드 완료')

# 윈도우 크기
screen_width, screen_height = 1280, 720

# 이미지 리사이즈 (좌우 2개)
img_w, img_h = screen_width // 2, screen_height
resized_images = [cv2.resize(img, (img_w, img_h)) for img in images]

# 얼굴 감지 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)  # refine 필수!

# 웹캠
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # 배경 이미지 합치기 (좌우)
    display = cv2.hconcat([resized_images[0], resized_images[1]])

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            # 왼쪽/오른쪽 눈동자 중심 좌표
            h, w, _ = frame.shape
            left_eye = face.landmark[468]
            right_eye = face.landmark[473]

            eye_x = (left_eye.x + right_eye.x) / 2
            eye_y = (left_eye.y + right_eye.y) / 2

            # 눈 좌표를 스크린 좌표로 변환
            cx = int(eye_x * screen_width)
            cy = int(eye_y * screen_height)

            # 작은 움직임 확대 (민감도 조절)
            center_x = screen_width // 2
            center_y = screen_height // 2
            amplify = 2.5  # 민감도: 값이 클수록 민감
            cx = int(center_x + (cx - center_x) * amplify)
            cy = int(center_y + (cy - center_y) * amplify)

            # 점 표시
            cv2.circle(display, (cx, cy), 10, (0, 255, 0), -1)

            # 시선이 향한 그림 판단
            grid_x = cx // img_w
            idx = grid_x
            if 0 <= idx < 2:
                cv2.putText(display, f"Looking at: 그림 {idx+1}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Art Eye Tracker", display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
