import cv2
import mediapipe as mp
import time
import os


def get_path():
    file_path = os.path.abspath(__file__)
    file_name = os.path.basename(__file__)
    current_path = file_path.replace(file_name, "")
    return current_path


def select_folder_dialog():
    current_path = get_path()
    folder_path = f"(current_path)/save_1/"
    return folder_path


current_path = get_path()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(f'{current_path}/audience.mp4')

# Initialize the Mediapipe face detection module
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    num_detection = 0  # Initialize the counter for the number of detections
    prev_faces = []  # List to store the previously detected face coordinates

    start_time = time.time()  # Start measuring the execution time

    face_crop_folder = select_folder_dialog()  # Select the face crop saving folder

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

                    # Save the face crop in the selected folder
                    face_crop_path = os.path.join(
                        face_crop_folder, f'face_crop_{num_detection}.png')
                    cv2.imwrite(face_crop_path, face_crop)

                    # Create a separate window for each face crop and move them to specific positions
                    cv2.imshow(f'face_crop_{num_detection}', face_crop)
                    # Adjust the position as per your preference
                    cv2.moveWindow(
                        f'face_crop_{num_detection}', 200 + num_detection * 70, 50)

                    # Add the current face coordinates to the list of previously detected faces
                    prev_faces.append((xmin, ymin, width, height))

                # Detecting the face 2000 times and break
                else:
                    consecutive_duplicates += 1
                    if consecutive_duplicates >= 2000:
                        break

        frame_width = int(frame.shape[1] * 0.5)
        frame_height = int(frame.shape[0] * 0.5)
        cv2.imshow('video', cv2.resize(frame, (frame_width, frame_height)))
        cv2.moveWindow('video', 300, 200)

        if cv2.waitKey(10) == ord('q'):  # exit when 'q' is pressed
            break

        if consecutive_duplicates >= 2000:
            break

    end_time = time.time()  # Stop measuring the execution time

cap.release()
cv2.destroyAllWindows()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
print("Number of detected people:", num_detection)
