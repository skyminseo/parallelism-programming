import cv2
import mediapipe as mp
import time
from multiprocessing import Process, Queue
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


def detect_faces(video_file, face_crop_folder, result, process_id):
    cap = cv2.VideoCapture(video_file)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the Mediapipe face detection module
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        num_detection = 0  # Initialize the counter for the number of detections
        prev_faces = []  # List to store the previously detected face coordinates

        while True:
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
                        int(bbox.width *
                            frame.shape[1]), int(bbox.height * frame.shape[0])

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
                        face_crop = frame[ymin:ymin +
                                          height, xmin:xmin + width]

                        face_crop_path = os.path.join(
                            face_crop_folder, f'face_crop_{num_detection}.png')
                        cv2.imwrite(face_crop_path, face_crop)

                        cv2.imshow(
                            f'face_crop_{process_id}_{num_detection}', face_crop)

                        cv2.moveWindow(
                            f'face_crop_{process_id}_{num_detection}', 10 + process_id * 100, 10 + num_detection * 60)

                        # Add the current face coordinates to the list of previously detected faces
                        prev_faces.append((xmin, ymin, width, height))

                    else:
                        consecutive_duplicates += 1
                        if consecutive_duplicates >= 2000:
                            break

            # position of each frame
            frame_width = int(frame.shape[1] * 0.2)
            frame_height = int(frame.shape[0] * 0.2)

            cv2.imshow(f'video_0', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_0', 800, 3)

            cv2.imshow(f'video_1', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_1', 800, frame_height + 3)

            cv2.imshow(f'video_2', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_2', 800, 2 * (frame_height + 3))

            cv2.imshow(f'video_3', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_3', 800, 3 * (frame_height + 3))

            cv2.imshow(f'video_4', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_4', 800, 4 * (frame_height + 3))

            cv2.imshow(f'video_5', cv2.resize(
                frame, (frame_width, frame_height)))
            cv2.moveWindow(f'video_5', 800, 5 * (frame_height + 3))

            if cv2.waitKey(10) == ord('q'):  # exit when 'q' is pressed
                break

            if consecutive_duplicates >= 2000:
                break

        cap.release()
        cv2.destroyAllWindows()

        result.put(num_detection)


if __name__ == '__main__':
    current_path = get_path()

    video_files = ['a1.mp4', 'a2.mp4', 'a3.mp4', 'a4.mp4', 'a5.mp4', 'a6.mp4']

    result_queue = Queue()
    processes = []

    start_time = time.time()  # Start measuring the execution time

    # Select the face crop saving folder
    face_crop_folder = select_folder_dialog()

    # Create and start a process for each video file
    for i, video_file in enumerate(video_files):
        process = Process(target=detect_faces, args=(
            f'{current_path}/{video_file}', face_crop_folder, result_queue, i))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    end_time = time.time()  # Stop measuring the execution time

    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    total_detection = 0
    while not result_queue.empty():
        num_detection = result_queue.get()
        total_detection += num_detection

    print("Total number of detected people:", total_detection)
