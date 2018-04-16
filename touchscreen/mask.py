import cv2
import numpy as np


def findContours(mask, original):
    """
        * finds all the contours on the image color mask
        * to the five largest contours in terms of the area,
            the minimum enclosing circle is fit and added to the image
    """

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        for cnt in cnts:
            # c = max(cnts, key=cv2.contourArea)
            # c = cnt
            rect = cv2.minAreaRect(cnt)
            print rect
            x, y, endx, endy = int(rect[0][0]), int(
                rect[0][1]), int(rect[1][0]), int(rect[1][1])

            cv2.rectangle(original, (x, y), (endx, endy), (0, 255, 0), 2)

        return original

    return original


def getMask(hsv):
    """
        This method masks all the colors.
        Pass the color as a parameter with the image in hsv format.
    """

    lower = np.array([00, 100, 50])
    upper = np.array([18, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask
