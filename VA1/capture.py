import numpy as np
import cv2
import cython
import argparse
import cv2
import time

import pyximport
pyximport.install()

import utils


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

    sensitivity = 30

    lower = np.array([0, 100, 50])
    upper = np.array([0, 255, 255])

    if color == 'blue':
        lower[0] = blue_hsv[0][0][0] - sensitivity
        upper[0] = blue_hsv[0][0][0] + sensitivity
    elif color == 'green':
        lower[0] = green_hsv[0][0][0] - sensitivity
        upper[0] = green_hsv[0][0][0] + sensitivity
    else:
        lower[0] = 169  # red_hsv[0][0][0]
        upper[0] = 189  # red_hsv[0][0][0] + sensitivity

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def getCircles(original, speed, single):
    """
        ALGORITHM:

        * converts the image to HSV format from BGR
        * creates a mask of the GREEN color
        * the mask is used to identify contours which are then used to find circular objects in the image
        * returns the image with the circle

    """

    if speed == 'slow':
        hsv = np.array(utils.bgr2hsv(original))
    else:
        hsv = bgr2hsv(original)

    # check = hsv - bgr2hsv(original)

    # if np.array(check).any() > 0:
    #     print check
    #     print "NOT RIGHT"

    green = getMask(hsv, 'green')

    green_circle = findContourCenter(green, original, single)

    return green_circle


def detectCircle(image, original):
    """
        Method detects circular objects in a grayscale image
        returns the output with the circle drawn over the circle
        else returns None if no circle is found.
    """

    output = original.copy()

    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT, 1.4, 100, param1=20, param2=100)

    if circles is not None:
        circles = np.around(circles[0, :].astype('int'))

        for (x, y, r) in circles:

            cv2.circle(output, (x, y), r, (255, 0, 0), 4)
            cv2.circle(original, center, 5, (0, 0, 255), -1)

        return output

    return None


def euclidean(x, y):
    """
        calculates euclidean distance between two points in a 2d space
    """
    return np.sqrt(np.sum(((x[0] - y[0])**2) + ((x[1] - y[1])**2)))


def findContourCenter(mask, original, single):
    """
        * finds all the contours on the image color mask
        * to the five largest contours in terms of the area,
            the minimum enclosing circle is fit and added to the image
    """

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    centers = []

    if len(cnts) > 0 and not single:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for cnt in cnts:
            # c = max(cnts, key=cv2.contourArea)
            c = cnt
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers.append((center, radius, cv2.contourArea(c)))

        for index, value in enumerate(centers):
            center, radius, area_value = value
            distances = [euclidean(center, c) > r for c,
                         r, a in centers if area_value > a]

            if np.array(distances).any() == False:
                centers.remove(value)

        for center, radius, area_value in centers:
            if radius > 20:
                cv2.circle(original, (int(center[0]), int(center[1])), int(radius),
                           (255, 0, 0), 2)
                cv2.circle(original, center, 2, (0, 0, 255), -1)

        return original

    elif len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 20:
            cv2.circle(original, (int(center[0]), int(center[1])), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(original, center, 2, (0, 0, 255), -1)

        # print centers
        return original

    return None


def captureVideo(query, speed, outfile, fps, single=True):
    """
        ALGORITHM:

        * captures the video stream
        * calls the getCircles function to find all the circular object of a single color
        * stores the last found detection and returns it for further processing.

    """

    if query == 'capture':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(query)

    _, last_image = cap.read()
    (h, w) = last_image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outfile, fourcc, fps, (w, h))

    start = time.time()
    frameCount = 0
    while(True):
        ret, frame = cap.read()

        # finds all circles in image colored RGB
        circle = getCircles(frame, speed, single)

        if circle is not None:
            result = circle
            last_image = result
        else:
            result = frame

        out.write(result)
        frameCount += 1

        cv2.imshow('BALLS', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    print "Frames Per Second:", frameCount / (end - start)
    print "Number of Frames:", frameCount
    print "Time Noted:", (end - start)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return last_image


def clearImage(last_image):
    """
        Part 2 of the assignment

        * add noise to the image
        * remove noise from the image
        * display both images
    """

    cv2.imwrite("ImageSelected.jpg", np.array(last_image))

    t = time.time()
    noisy = addNoise(last_image)
    print "TIME for noise addition:", time.time() - t
    # cv2.imshow("NOISY DETECTION", noisy)
    cv2.imwrite("Noisy.jpg", np.array(noisy))

    t = time.time()
    remove = removeNoise(noisy, 5)
    print "TIME for noise removal:", time.time() - t

    # cv2.imshow("NOISE REMOVAL", np.array(remove))
    cv2.imwrite("NoiseRemoved.jpg", np.array(remove))
    # cv2.waitKey(0)
    print "All Done."
    # cv2.destroyAllWindows()

    return


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
