import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
facedectection = mpFaceDetection.FaceDetection()

while True:
    success , img = cap.read()
    imgrgb = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    results = facedectection.process(imgrgb)


    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(1)