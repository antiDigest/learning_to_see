
"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import numpy as np
import cv2
import time

windows = [(64, 128), (128, 128), (128, 64)]


# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh, ratios=None):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    if ratios == None:
        return boxes[pick].astype("int")
    return boxes[pick].astype("int"), ratios[pick]


def pyramid(image, scale=1.5, minSize=(20, 20)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))  # imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def capture():
    (winW, winH) = (256, 256)
    cap = cv2.VideoCapture(0)

    while(1):
        ret, frame = cap.read()

        # loop over the image pyramid
        for resized in pyramid(frame):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore
                # it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                # WINDOW

                # since we do not have a classifier, we'll just draw the window
                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + winW,
                                              y + winH), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                # time.sleep(0.025)

        cv2.imshow('SLIDING WINDOW', frame)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
