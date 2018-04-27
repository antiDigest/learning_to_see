import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('face.xml')


def detect_face(image):

    faces = faceCascade.detectMultiScale(
        image, scaleFactor=1.15, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)

    return image


def video_capture():

    cap = cv2.VideoCapture(0)

    while(1):
        # for frame in imgs:
        _, frame = cap.read()

        # frame = cv2.imread(frame)

        image = frame.copy()

        image = detect_face(image)

        cv2.imshow("Face detection", image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    video_capture()
