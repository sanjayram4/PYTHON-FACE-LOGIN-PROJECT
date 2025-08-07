import cv2
import face_recognition
import numpy as np
import os
from playsound import playsound

# Load known face
known_face_path = 'Known_faces'
known_image = face_recognition.load_image_file(known_face_path)
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load pre-trained gender classifier (from OpenCV)
gender_model = cv2.dnn.readNetFromCaffe(
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_gender.prototxt',
    'https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/gender_net.caffemodel?raw=true'
)
GENDER_LIST = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    return GENDER_LIST[gender_preds[0].argmax()]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"[INFO] Total faces detected: {len(face_encodings)}")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        face_img = frame[top:bottom, left:right]
        label = "Unknown"

        if True in matches:
            label = "OK YOU WILL GO"
            color = (0, 255, 0)
        else:
            label = "ALERT! UNKNOWN PERSON"
            color = (0, 0, 255)
            playsound("alarm.mp3", block=False)

        # Try to detect gender
        try:
            gender = detect_gender(face_img)
        except:
            gender = "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{label} - {gender}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display count
    cv2.putText(frame, f"Total People: {len(face_encodings)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Face Login System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()