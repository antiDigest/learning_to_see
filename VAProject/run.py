
"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import cv2
import numpy as np
from person import *
from face import *
from eyes import *
from logo_template import *
from logo import *


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 0, 255)
lineType = 1
bottomLeftCornerOfText = (10, 50)


def euclidean(x, y):
    """
        calculates euclidean distance between two points in a 2d space
    """
    return np.sqrt(np.sum(((x[0] - y[0])**2) + ((x[1] - y[1])**2)))


def in_region(region, face):
    (xA, yA, xB, yB) = region
    (x, y, w, h) = face
    return (xA <= x and xB >= x + w and yA <= y and yB >= y + h)


def detect_face_and_eyes(image, region):
    (xA, yA, xB, yB) = region
    faces = detect_face(image)
    for face in faces:
        (x, y, w, h) = face
        if in_region(region, face):
            face = image[y:y + h, x:x + w]
            eyes = detect_eyes(image)
            region_face = (xA + x, yA + y, xA + w, yA + h)
            for eye in eyes:
                (xe, ye, we, he) = eye
                if in_region(region_face, eye):
                    cv2.rectangle(image, (xe, ye), (xe + we, ye + he),
                                  (255, 0, 255), 3)
            cv2.rectangle(image, (x, y),
                          (x + w, y + h), (69, 165, 255), 2)

    return image


def detect():

    cap = cv2.VideoCapture(0)

    self_height = 0.
    height = 0.

    while(1):
        _, frame = cap.read()
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        clone = image.copy()

        image, rects = detect_person(image)

        founds = []
        for rect in rects:
            (xA, yA, xB, yB) = rect
            window = clone[yA:yB, xA:xB]
            found = match(window)
            founds.append([found[0], found[1], found[2], rect])

        sorted_found = sorted(founds, key=lambda tup: tup[0], reverse=True)

        for index, found in enumerate(sorted_found):
            (xA, yA, xB, yB) = found[3]
            if index == 0:
                self_height = euclidean([xA, yA], [xA, yB])
                ratio = 172. / self_height
                image = detect_face_and_eyes(image, (xA, yA, xB, yB))
                cv2.rectangle(image, (xA, yA),
                              (xB, yB), (0, 255, 0), 2)
            else:
                height = euclidean([xA, yA], [xA, yB])

                cv2.rectangle(image, (xA, yA),
                              (xB, yB), (0, 0, 255), 2)

        if self_height != 0. and height != 0.:
            actualHeight = ratio * height
            cv2.putText(image, str(actualHeight) + " cm",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        cv2.imshow('Person with Logo Shirt', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # init()

    detect()
