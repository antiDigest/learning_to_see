import numpy as np
import cv2
import cython
import argparse
import cv2
import time

import pyximport
pyximport.install()

pi = 22. / 7.


def euclidean(x, y):
    """
        calculates euclidean distance between two points in a 2d space
    """
    return np.sqrt(np.sum(((x[0] - y[0])**2) + ((x[1] - y[1])**2)))


def findContourCenter(mask, original, centers):
    """
        * finds all the contours on the image color mask
        * to the five largest contours in terms of the area,
            the minimum enclosing circle is fit and added to the image
    """

    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # ((x, y), radius) = cv2.minEnclosingCircle(c)
    # cv2.drawContours(original, cnts, -1, (0, 255, 0), 1)
    center = None

    if len(cnts) > 0:
        for cnt in cnts:
            # c = min(cnts, key=cv2.contourArea)
            c = cnt
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if len(centers) > 1 and \
                    euclidean(center, centers[-1]) > (euclidean(centers[-2], centers[-1]) + 20):
                continue

            approx = cv2.arcLength(c, True)
            area = 4. * pi * cv2.contourArea(c)
            circularity = area / approx**2.

            print circularity, radius
            if ((circularity > 0.79)) and radius > 5:
                # contour_list.append(contour)
                cv2.circle(original, (int(center[0]), int(center[1])), int(radius),
                           (255, 0, 0), 2)
                cv2.circle(original, center, 2, (0, 0, 255), -1)

                # print centers
                centers.append(center)
                return original, centers

    return None, centers


def drawPath(centers, original):
    """
        draws a path between all pairs of points in the centers list
    """

    if len(centers) > 0:
        l = len(centers) - 1
        for i in range(1, l + 1):
            cv2.line(original, centers[i - 1], centers[i], (255, 255, 255))

    return original


def writeSpeed(centers, original, numFrames, fps):
    """
        Writes the speed on top left corner of the video file.
        * Speed is written in pixels per second
    """

    l = len(centers) - 1

    pixels = euclidean(centers[l - 1], centers[l])

    speed = pixels * fps / numFrames
    text = "{0:.3f}".format(speed)
    cv2.putText(original, str(text) + " p/s", (300, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    return original, speed


def captureVideo(query):
    """
        ALGORITHM:

        * captures the video stream
        * subtracts the previous frame (background) to find the moving object
        * calculates the threshold of the resulting image
        * checks if the thresholded image has a moving object
        * if yes, checks if the object is a ball
        * if yes, marks the ball and tracks its center
        * center is used for drawing path

    """

    if query == 'capture':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(query)
        numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

    print fps

    background = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    black = cv2.subtract(background, background)

    centers = []
    start = time.time()
    frameCount = 0
    avgSpeed = 0.
    while(True):
        ret, frame = cap.read()

        frameCount += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        frameDelta = cv2.subtract(gray, background)
        thresh = cv2.threshold(frameDelta, 65, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        circle, centers = findContourCenter(thresh, frame, centers)
        if circle is not None:
            result = circle
        else:
            result = frame

        if len(centers) > 1:
            path = drawPath(centers, black)
            if circle is not None:
                result, speed = writeSpeed(
                    centers, result, frameCount - framesTillNow, fps)
                avgSpeed += speed
        else:
            path = black
        framesTillNow = frameCount

        background = gray

        cv2.imshow('Motion Detection', result)
        cv2.imshow('Path', path)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or frameCount == int(numFrames) - 1:
            break

    end = time.time()
    print "Frames Per Second:", frameCount / (end - start)
    print "Number of Frames:", frameCount
    print "Average speed of the ball: ", "{0:.2f}".format((avgSpeed) / (end - start)), "p/s"
    print "Time Noted:", (end - start)

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
        ****** ASSIGNMENT 2 ******

        This part calls two halves of the assignment

        * Trace a path of the ball in question

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", default="capture",
                    help="Path to the query video ('capture' for VideoCapture from webcam)")
    # ap.add_argument("-n", "--number", type=bool, default=True,
    # help="Number of balls to detect (Multiple balls (False) or single ball
    # (True))")
    args = vars(ap.parse_args())

    # print args['number']

    captureVideo(args['query'])


if __name__ == '__main__':
    main()
