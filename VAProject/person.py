
"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import cv2
import numpy as np
from utils import *
import time
from face import *

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imgs = ['data/find1(9).jpg', 'data/find1(10).jpg']


def detect_person(image):
    (rects, weights) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(32, 32), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    # print "Pedestrians: ", len(rects)
    pick = non_max_suppression(rects, overlapThresh=0.5)

    return image, pick


def video_capture():

    cap = cv2.VideoCapture(0)

    while(1):
        # for frame in imgs:
        _, frame = cap.read()

        # frame = cv2.imread(frame)

        image = frame.copy()

        image = detect_person(image)

        cv2.imshow("Person detection", image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    video_capture()
