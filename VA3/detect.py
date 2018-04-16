import numpy as np
import cv2
import cython
import argparse
import cv2
import time

import pyximport
pyximport.install()


PI = 3.141527


def bgr2hsv(bgr):
    """
        Quick BGR to HSV conversion
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def addNoise(img, density=0.2):
    """
        Adds salt and pepper noise to the image
        * randomly generates a 2d array of numbers of the size of the image
        * finds all coordinates where the numbers in the 2d array are less than density
            and assigns them the value 0 (pepper)
        * finds all coordinates where the numbers in the 2d array are greater than density
            and assigns them the value 255 (salt)
    """

    # img = img.astype('float')
    I = img.copy()
    J = I
    p = density

    x = np.random.rand(I.shape[0], I.shape[1])

    coords = np.where(x < (p / 2.))
    # print coords
    J[coords] = 0  # Minimum value
    coords = np.where(x > (1. - (p / 2.)))
    J[coords] = 255  # Maximum(saturated) value

    # J = cv2.cvtColor(J, cv2.COLOR_GRAY2BGR)
    return J


def removeNoise(img, size):

    I = img.copy()

    # G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    return utils.medianFilter(I, size)


def getMask(hsv, color):
    """
        This method masks all the colors.
        Pass the color as a parameter with the image in hsv format.
    """

    blue = np.uint8([[[255, 0, 0]]])
    blue_hsv = np.array(bgr2hsv(blue))
    green = np.uint8([[[0, 255, 0]]])
    green_hsv = np.array(bgr2hsv(green))
    red = np.uint8([[[0, 0, 255]]])
    red_hsv = np.array(bgr2hsv(red))

    sensitivity = 18

    lower = np.array([0, 100, 50])
    upper = np.array([0, 255, 255])

    if color == 'blue':
        lower[0] = blue_hsv[0][0][0] - sensitivity
        upper[0] = blue_hsv[0][0][0] + sensitivity
    elif color == 'green':
        lower[0] = green_hsv[0][0][0] - sensitivity
        upper[0] = green_hsv[0][0][0] + sensitivity
    else:
        lower[0] = red_hsv[0][0][0] - sensitivity
        upper[0] = red_hsv[0][0][0] + sensitivity

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=5)

    return mask


def getCircles(original, centers):
    """
        ALGORITHM:

        * converts the image to HSV format from BGR
        * creates a mask of the GREEN color
        * the mask is used to identify contours which are then used to find circular objects in the image
        * returns the image with the circle

    """

    hsv = bgr2hsv(original)

    # check = hsv - bgr2hsv(original)

    # if np.array(check).any() > 0:
    #     print check
    #     print "NOT RIGHT"

    green = getMask(hsv, 'green')
    red = getMask(hsv, 'red')
    blue = getMask(hsv, 'blue')

    # cv2.imshow("Mask", red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # circles, centers = findContourCenter(green, original, centers)
    circles, centers = findContourCenter(red, original, centers)
    # circles, centers = findContourCenter(blue, circles, centers)

    return circles, centers


def findContourCenter(mask, original, centers):
    """
        * finds all the contours on the image color mask
        * to the five largest contours in terms of the area,
            the minimum enclosing circle is fit and added to the image
    """

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        for cnt in cnts:
            # c = min(cnts, key=cv2.contourArea)
            c = cnt
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            approx = cv2.arcLength(c, True)
            area = 4. * PI * cv2.contourArea(c)
            circularity = area / approx**2.

            # print circularity, radius
            if ((circularity > 0.5) and radius > 10):
                # contour_list.append(contour)
                centers.append((x, y))
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return original, centers


def main():
    """
        ****** ASSIGNMENT 1 ******

        This part calls two halves of the assignment

        * captureVideo runs the video analysis of tracking a ball
        * clearImage runs the image analysis

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", default="capture",
                    help="Path to the query video ('capture' for VideoCapture from webcam)")
    ap.add_argument("-s", "--speed", default="slow",
                    help="""Which conversion would you prefer ('slow'
                         runs self implemented bgr2hsv and 'fast' runs using cvtColor)""")
    ap.add_argument("-o", "--output", default="video/capture.avi",
                    help="path to output video file (default: 'video/capture.avi')")
    ap.add_argument("-f", "--fps", type=int, default=8,
                    help="FPS of output video (default: 8)")
    # ap.add_argument("-n", "--number", type=bool, default=True,
    # help="Number of balls to detect (Multiple balls (False) or single ball
    # (True))")
    args = vars(ap.parse_args())

    # print args['number']

    last_image = captureVideo(args['query'], args['speed'], args[
                              'output'], args['fps'])

    clearImage(last_image)


if __name__ == '__main__':
    main()
