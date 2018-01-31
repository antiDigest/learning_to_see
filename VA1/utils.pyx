import cython
import numpy as np
import cv2


@cython.boundscheck(False)
cpdef double getHue(double cmax, double r, double g, double b, double delta):
    """
        produces the value of HUE
    """

    cdef double h

    if delta == 0:
        return 0.
    h = 0.

    if (cmax == r):
        h = (abs(g - b) / (delta) % 6.);
    elif (cmax == g):
        h = 2. + abs(b - r) / (delta);
    else:
        h = 4. + abs(r - g) / (delta);

    h *= 60.
    if h < 0:
        h += 360.

    return h

@cython.boundscheck(False)
cpdef unsigned char [:] getHSV(unsigned char [:] bgr):
    """
        Takes in pixel as BGR and returns HSV format pixel
    """

    cdef double cmax, cmin, delta, r, g, b
    hsv = bgr.copy()

    # bgr = bgr.astype('float16')
    r = float(bgr[2]) / 255.
    g = float(bgr[1]) / 255.
    b = float(bgr[0]) / 255.

    cmax = max(r, max(g, b))
    cmin = min(r, min(g, b))
    v = cmax * 255

    delta = cmax - cmin
    if v == 0.:
        # print "TRUE"
        s = 0
        h = 0
        hsv[0] = int(h)
        hsv[1] = int(s)
        hsv[2] = int(v)

        return hsv
    
    s = 255 * delta / cmax;
    if s == 0:
        h = 0
        hsv[0] = int(h)
        hsv[1] = int(s)
        hsv[2] = int(v)

        return hsv

    h = getHue(cmax, r, g, b, delta) * (1. / 2.)
    
    hsv[0] = int(h)
    hsv[1] = int(s)
    hsv[2] = int(v)

    return hsv

@cython.boundscheck(False)
cpdef unsigned char [:, :, :] bgr2hsv(unsigned char [:, :, :] bgr):
    """

        converts the image from BGR type to HSV

        ****** SELF IMPLEMENTATION ******

        returns image converted to hsv

    """

    cdef int h, w, i, j

    h = bgr.shape[0]
    w = bgr.shape[1]

    hsv = bgr.copy()

    for i in range(0, w):
        for j in range(0, h):
            # print getHSV(bgr[j, i])
            hsv[j, i] = getHSV(bgr[j, i])

    # return hsv

    return hsv


@cython.boundscheck(False)
cpdef unsigned char [:, :, :] medianFilter(unsigned char [:, :, :] image, int size):
    """
        Median Filter: works great with salt and pepper noise
        My code to remove salt and pepper noise
    """

    cdef int h, w, ch, i, j
    cdef unsigned char [:, :, :] I

    I = image.copy()
    image = cv2.copyMakeBorder(np.array(image), size/2 + 1, size/2 + 1, size/2 + 1, size/2 + 1, cv2.BORDER_REPLICATE)

    h = I.shape[0]
    w = I.shape[1]

    for i in range(0, h):
        for j in range(0, w):
            I[i, j, 0] = np.median(
                image[i: i + size, j: j + size, 0])
            I[i, j, 1] = np.median(
                image[i: i + size, j: j + size, 1])
            I[i, j, 2] = np.median(
                image[i: i + size, j: j + size, 2])

    return I