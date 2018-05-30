
"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import cv2
import numpy as np
from utils import *
import time

faceCascade = cv2.CascadeClassifier('models/face.xml')


def detect_face(image):

    # start = time.time()
    faces = faceCascade.detectMultiScale(
        image, scaleFactor=1.05, minNeighbors=5)
    # print "Face Time: ", time.time() - start
    faces = non_max_suppression(faces, overlapThresh=0.5)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)

    return faces


def video_capture():

    cap = cv2.VideoCapture(0)

    while(1):
        # for frame in imgs:
        _, frame = cap.read()

        # frame = cv2.imread(frame)
        image = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        image = detect_face(image)

        cv2.imshow("Face detection", image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    video_capture()
