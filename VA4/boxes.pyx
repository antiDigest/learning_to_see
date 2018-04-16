%%cython -a
 
import numpy as np
import cv2
import cython
 
@cython.boundscheck(False)
cpdef unsigned char[:, :] checkBox(unsigned char [:, :]frame, int width, int height):

	cdef int x, y, w, h

    h, w, _ = frame.shape

    boxPreds = []
    for x in range(1, w - width, width / 5):
        for y in range(1, h - height, height / 5):
            boxPred = {}
            im = frame[x:x + width, y:y + height]
            kp, des = surf.detectAndCompute(im, None)
            histogram = bow_ext.compute(im, kp)
            row = rf.getVotes(histogram, 0)
            # print row.shape

            if row.shape == (1, 2):
                continue

            row = row[1]
            # print row
            if row[0] > row[1]:
                boxPred['value'] = row[0]
                boxPred['frame'] = frame
                boxPred['kp'] = kp
                boxPred['box'] = (x, y, width, height)
                boxPreds.append(boxPred)

    boxPred = sorted(boxPreds, key=lambda x: x['value'], reverse=True)

    return boxPred