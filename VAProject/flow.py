import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame2 = cap.read()

    n = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prvs, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.resizeWindow('FLOW', 600, 300)
    cv2.imshow('FLOW', np.hstack([mag, bgr, frame2]))

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opticalhsv.png', bgr)
        cv2.imwrite('opticalmag.png', mag)

    prvs = n

cap.release()
cv2.destroyAllWindows()
