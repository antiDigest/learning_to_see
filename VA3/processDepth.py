import numpy as np
import cv2
import cython
import argparse
import cv2
import time

import pyximport
pyximport.install()


def pixelTransform(pixel, kernel):
    return np.matmul(pixel, kernel)


def colorTransform(color, depthPixel):

    alternate = [0, 0, 0]

    if depthPixel[0] < 0 or depthPixel[1] < 0:
        return alternate
    if depthPixel[0] >= color.shape[1] or depthPixel[1] >= color.shape[0]:
        return alternate

    return color[int(depthPixel[1]), int(depthPixel[0])]


def depthToRgb(image, invIr, tdc, iRgb, color):
    h, w = image.shape

    colorized = cv2.resize(color.copy(), (w, h))
    dToRgb = colorized.copy()

    for x in range(0, w):
        for y in range(0, h):
            p1 = np.multiply(image[y, x], [x, y, 1])
            p1 = pixelTransform(p1, invIr)

            p2 = pixelTransform(p1, tdc[:3, :3]) + tdc[3, :3]

            p3 = pixelTransform(p2, iRgb) / p2[2]

            colorized[y, x] = colorTransform(color, np.rint(p3))

    return colorized


def processImage(query, folder):
    """
        ALGORITHM:

    """

    with open(folder + 'IntrinsicRGB', 'r') as f:
        intrinsicRGB = np.array(
            [el.split(',') for el in f.read().split('\n') if el != ''], dtype='float32')

    with open(folder + 'InvIntrinsicIR', 'r') as f:
        invIntrinsicIR = np.array(
            [el.split(',') for el in f.read().split('\n') if el != ''], dtype='float32')

    with open(folder + 'TransformationD-C', 'r') as f:
        transformationDC = np.array(
            [el.split(',') for el in f.read().split('\n') if el != ''], dtype='float32')

    depth = folder + "depth-" + query
    color = folder + "color-" + query

    depthImage = np.float32(cv2.imread(depth, cv2.IMREAD_ANYDEPTH))
    colorImage = cv2.imread(color)

    h, w = depthImage.shape

    start = time.time()

    colorized = depthToRgb(depthImage, invIntrinsicIR,
                           transformationDC, intrinsicRGB, colorImage)

    end = time.time()
    print "Transformation Time: ", end - start

    cv2.imshow('Colorized Depth Image', colorized)
    cv2.imwrite("colorizedDepthImage.png", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
        ****** ASSIGNMENT 3 ******

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", help="Path to the query image")
    ap.add_argument("-d", "--data", default="data/",
                    help="Path to the data folder")
    # ap.add_argument("-n", "--number", type=bool, default=True,
    # help="Number of balls to detect (Multiple balls (False) or single ball
    # (True))")
    args = vars(ap.parse_args())

    # print args['number']

    processImage(args['query'], args['data'])


if __name__ == '__main__':
    main()
