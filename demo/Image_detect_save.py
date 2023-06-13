import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('audience.mp4')

# Initialize the Mediapipe face detection module
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    num_detection = 0  # Initialize the counter for the number of detections
    prev_faces = []  # List to store the previously detected face coordinates

    while cap.isOpened() and num_detection < 30:  # Break the loop after 30 detections
        success, frame = cap.read()
        if not success:
            break

        target_frame = frame.copy()  # copy frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(
            gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        COLOR = (0, 200, 0)  # green

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

                    # Crop the card region based on the face bounding box
                    card_crop = frame[ymin:ymin + height, xmin:xmin + width]
                    cv2.imshow(f'card_crop_{num_detection}', card_crop)
                    cv2.imwrite(f'card_crop_{num_detection}.png', card_crop)

                    # Add the current face coordinates to the list of previously detected faces
                    prev_faces.append((xmin, ymin, width, height))

        cv2.imshow('frame', frame)
        cv2.imshow('contour', target_frame)

        if cv2.waitKey(1) == ord('q'):  # exit when 'q' is pressed
            break

cap.release()
cv2.destroyAllWindows()

print("Number of detections:", num_detection)
