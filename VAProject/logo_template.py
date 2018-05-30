

"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import numpy as np
import cv2
from glob import glob
from utils import *
from logo import *

surf = cv2.xfeatures2d.SURF_create()
template = cv2.imread('data/template.jpg')
template = cv2.resize(template, (0, 0), fx=0.2, fy=0.2)
tH, tW, _ = template.shape
windows = [(template.shape[0], template.shape[1])]
template_kp, template_des = surf.detectAndCompute(template, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

imgsWow = ['data/find1(9).jpg', 'data/find1(10).jpg', 'data/find1(11).jpeg']


def match_template(image):
    kp, des = surf.detectAndCompute(image, None)

    matches = []
    if len(kp) >= 2:
        matches = flann.knnMatch(template_des, des, k=2)

    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.55 * n.distance:
            good.append(m)

    return len(good)


def match_edge_template(image):
    template = cv2.imread('data/template.jpg')
    template = cv2.resize(template, (0, 0), fx=0.05, fy=0.05)
    template_edge = cv2.Canny(template, 50, 400)
    tH, tW, _ = template.shape

    found = None
    for scale in np.linspace(1.0, 1.5, 5)[::-1]:
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        r = image.shape[1] / float(resized.shape[1])
        edge = cv2.Canny(resized, 50, 400)
        result = cv2.matchTemplate(edge, template_edge, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable

        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # print startX, endX, startY, endY
        if image.shape[0] >= endX and image.shape[1] >= endY:
            # print type(image[startX: endX, startY: endY]), (image[startX:
            # endX, startY: endY]).dtype
            maxV = match_template(image[startX: endX, startY: endY])
            # print "Match Count:", maxV
            if found is None or maxV > found[0]:
                found = (maxV, maxLoc, r)

    return found


def match(image):

    image = cv2.medianBlur(image, 3)
    found = match_edge_template(image)

    if found == None:
        return (0., 0., 0.)
    return found


def video_capture():

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        image = cv2.resize(frame, (0, 0), fx=0.55, fy=0.55)
        frame = cv2.medianBlur(image, 5)

        found = match_edge_template(frame)

        if found != None:
            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (
                int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

        cv2.imshow("Matches", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':

    image = cv2.imread('data/find1(10).jpg')
    video_capture()
    match(image)
