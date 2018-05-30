
"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import cv2
import numpy as np
from utils import *
import time

eyeCascade = cv2.CascadeClassifier('models/eyes.xml')


def detect_eyes(image):

    image = cv2.resize(image, (0, 0), fx=4, fy=4)

    # start = time.time()
    eyes = eyeCascade.detectMultiScale(
        image, scaleFactor=2.5, minNeighbors=5)
    # print "Eye Time: ", time.time() - start
    eyes = non_max_suppression(eyes, overlapThresh=0.5)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (69, 165, 255), 2)

    return eyes


def video_capture():

    cap = cv2.VideoCapture(0)

    while(1):
        # for frame in imgs:
        _, frame = cap.read()

        # frame = cv2.imread(frame)
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        eyes = detect_eyes(image)
        for eye in eyes:
            (xe, ye, we, he) = eye
            cv2.rectangle(image, (xe, ye), (xe + we, ye + he),
                          (255, 0, 255), 3)

        cv2.imshow("Eye detection", image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    video_capture()
