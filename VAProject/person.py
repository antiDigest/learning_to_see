import cv2
import numpy as np
from utils import *
import time
from face import *

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imgs = ['data/find1(9).jpg', 'data/find1(10).jpg']


def detect_person(image):

    # detect people in the image
    start = time.time()
    (rects, weights) = hog.detectMultiScale(
        image, winStride=(6, 6), scale=1.05)
    print "Detection time:", time.time() - start

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, overlapThresh=0.65)

    # detect face and draw the bounding boxes
    for (xA, yA, xB, yB) in pick:
        window = image[xA:xB, yA:yB]
        image = detect_face(window)

        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    return image


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
