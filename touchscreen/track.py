import numpy as np
import cv2
import time
from mask import *


def capture():

    cap = cv2.VideoCapture(0)

    start = time.time()
    while(True):
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = getMask(hsv)
        orig = findContours(thresh, frame)

        cv2.imshow('Face', orig)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    end = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()
