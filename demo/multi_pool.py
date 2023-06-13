from multiprocessing import Pool
import cv2
import mediapipe as mp
import time
import threading
import os

cv2.setNumThreads(1)

active_count = threading.active_count()


def get_path():
    file_path = os.path.abspath(__file__)
    file_name = os.path.basename(__file__)
    current_path = file_path.replace(file_name, "")
    return current_path


current_path = get_path()

active_count = threading.active_count()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def get_path():
    file_path = os.path.abspath(__file__)
    file_name = os.path.basename(__file__)
    current_path = file_path.replace(file_name, "")
    return current_path


current_path = get_path()

active_count = threading.active_count()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(f'{current_path}/audience.mp4')

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    num_detection = 0  # Initialize the counter for the number of detections
    prev_faces = []  # List to store the previously detected face coordinates

    start_time = time.time()  # Start measuring the execution time

    while True:  # Break the loop after 30 detections
        if num_detection > 30:
            break
        success, frame = cap.read()
        if not success:
            break

        target_frame = frame.copy()  # copy frame

        # Perform face detection using Mediapipe on the current frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)


def process_frame(frame):
    # Perform face detection using Mediapipe on the current frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw face detection annotations on the frame and crop card region
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

            # Get the bounding box coordinates of the detected face
            bbox = detection.location_data.relative_bounding_box
            xmin, ymin, width, height = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
                int(bbox.width * frame.shape[1]
                    ), int(bbox.height * frame.shape[0])

            # Check if the current face overlaps with any previously detected faces
            is_duplicate = False
            for prev_face in prev_faces:
                if abs(xmin - prev_face[0]) < prev_face[2] and abs(ymin - prev_face[1]) < prev_face[3]:
                    is_duplicate = True
                    break

            if not is_duplicate:
                num_detection += 1
                consecutive_duplicates = 0

                # Crop the face region based on the face bounding box
                face_crop = frame[ymin:ymin + height, xmin:xmin + width]

                cv2.imshow(f'face_crop_{num_detection}', face_crop)
                cv2.imwrite(f'face_crop_{num_detection}.png', face_crop)

                # Add the current face coordinates to the list of previously detected faces
                prev_faces.append((xmin, ymin, width, height))

            else:
                consecutive_duplicates += 1
                if consecutive_duplicates >= 2000:
                    break

    return frame


if __name__ == '__main__':
    pool = Pool(processes=8)
# Initialize the Mediapipe face detection module
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        num_detection = 0  # Initialize the counter for the number of detections
        prev_faces = []  # List to store the previously detected face coordinates

        start_time = time.time()  # Start measuring the execution time

        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)

        with Pool() as p:
            processed_frames = p.map(process_frame, frames)

        for frame in processed_frames:
            cv2.imshow('frame', cv2.resize(frame, None, fx=0.8, fy=0.8))

            if cv2.waitKey(1) == ord('q'):  # exit when 'q' is pressed
                break

        end_time = time.time()  # Stop measuring the execution time

    cap.release()
    cv2.destroyAllWindows()

    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Number of detected people:", num_detection)
    print("active thread", active_count)
