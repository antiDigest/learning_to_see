import numpy as np
import cv2
import cython
import argparse
import cv2
import time

from detect import *

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


def invTransform(pixel, invIr, tdc, iRgb):
    p1 = pixel[:] / iRgb

    p2 = p1[:] / tdc[:3, :3] - tdc[3, :3]

    p3 = p2[:] / invIr

    return p3


def transform(pixel, invIr, tdc, iRgb):

    p1 = pixelTransform(pixel, invIr)

    p2 = pixelTransform(p1, tdc[:3, :3]) + tdc[3, :3]

    p3 = pixelTransform(p2, iRgb) / p2[2]

    return p3


def depthToRgb(image, invIr, tdc, iRgb, color):
    h, w = image.shape

    colorized = cv2.resize(color.copy(), (w, h))
    dToRgb = colorized.copy()

    for x in range(0, w):
        for y in range(0, h):
            p1 = np.multiply(image[y, x], [x, y, 1])
            p3 = transform(p1, invIr, tdc, iRgb)
            colorized[y, x] = colorTransform(color, np.rint(p3))

    return colorized


def euclidean(x, y):
    """
        calculates euclidean distance between two points in a 2d space
    """
    return np.sqrt(np.sum(((x[0] - y[0])**2) + ((x[1] - y[1])**2)))


def processImage(queries, folder):
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

    images = []
    dist = []
    for query in queries:
        depth = folder + "depth-" + query
        color = folder + "color-" + query

        depthImage = np.float32(cv2.imread(depth, cv2.IMREAD_ANYDEPTH))
        colorImage = cv2.imread(color)

        h, w = depthImage.shape

        start = time.time()

        colorized = depthToRgb(depthImage, invIntrinsicIR,
                               transformationDC, intrinsicRGB, colorImage)

        cv2.normalize(colorized, colorized, 0, 255, cv2.NORM_MINMAX)

        end = time.time()
        print "Transformation Time: ", end - start

        image, centers = getCircles(colorized, [])

        c1 = transform(np.append(centers[0], 1), invIntrinsicIR,
                       transformationDC, intrinsicRGB)
        c2 = transform(np.append(centers[1], 1), invIntrinsicIR,
                       transformationDC, intrinsicRGB)
        # print c1, c2
        dist.append(euclidean(c1, c2))
        images.append(image)

    print "Relative Velocity:", np.absolute(dist[0] - dist[1]) / 1.300, "meters/second"

    cv2.imshow('Colorized Depth Image', np.hstack(images))
    # cv2.imwrite("colorizedDepthImage.png", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
        ****** ASSIGNMENT 3 ******

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", help="Path to the query images")
    ap.add_argument("-d", "--data", default="data/",
                    help="Path to the data folder")
    # ap.add_argument("-n", "--number", type=bool, default=True,
    # help="Number of balls to detect (Multiple balls (False) or single ball
    # (True))")
    args = vars(ap.parse_args())

    # print args['number']

    processImage(args['query'].split(','), args['data'])


if __name__ == '__main__':
    main()
