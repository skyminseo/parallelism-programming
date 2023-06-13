import cv2
import dlib

# 얼굴 정면 인식을 위한 Dlib의 얼굴 인식기 초기화
detector = dlib.get_frontal_face_detector()

def save_frontal_faces(video_file, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0  # 프레임 수 카운트 변수

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = detector(gray)

        for face in faces:
            # 얼굴 영역 추출
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = frame[y:y+h, x:x+w]

            # 이미지 파일로 저장
            output_file = f"{r'Users\\skymi\\python\\PythonImageWorkspace\\CW2\\output_faces}/face_{frame_count}.jpg"
            cv2.imwrite(output_file, face_img)

        frame_count += 1

    cap.release()

# 비디오 파일에서 정면 얼굴 인식하여 이미지 파일로 저장
save_frontal_faces('conversation.mp4', 'output_faces')
