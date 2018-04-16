import numpy as np
import cv2
from glob import glob
from utils import *


TRAIN = "train/"
TEST = "test/"

surf = cv2.xfeatures2d.SURF_create(extended=True)
bow_ext = cv2.BOWImgDescriptorExtractor(
    surf, cv2.BFMatcher(cv2.NORM_L2))
kmeansTrainer = cv2.BOWKMeansTrainer(80)
svm = cv2.ml.SVM_create()
rf = cv2.ml.RTrees_create()


def runCapture(clf):
    (winW, winH) = (64, 64)
    cap = cv2.VideoCapture(0)

    while(1):
        ret, frame = cap.read()

        # frame = cv2.imread(image)
        maxVote = 0
        found = None

        # Loop over the different window sizes
        for (winW, winH) in windows:
            # loop over the image pyramid
            for resized in pyramid(frame):
                # loop over the sliding window for each layer of the pyramid
                for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                    # if the window does not meet our desired window size, ignore
                    # it
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue

                    kp, des = surf.detectAndCompute(window, None)
                    histogram = bow_ext.compute(window, kp)
                    value = rf.predict(histogram)

                    # if value > max

                    # since we do not have a classifier, we'll just draw the
                    # window
                    if value[0] == 1.0:
                        votes = rf.getVotes(histogram, 0)

                        if (votes[1, 0] > maxVote):
                            maxVote = votes[1, 0]
                            found = (maxVote, (x, y, x + winW, y + winH),
                                     frame.shape[1] / resized.shape[1])

        if found != None:
            (vote, window, r) = found
            startx, starty, endx, endy = (w * r for w in window)
            cv2.putText(frame, "Temoc " + str(vote), (startx, starty),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))
            cv2.rectangle(frame, (startx, starty),
                          (endx, endy), (0, 255, 0), 2)

        cv2.imshow("Temoc", frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():

    train_y = []
    train_x = []

    desc = []

    for file in glob(TRAIN + "*"):
        print file
        frame = cv2.imread(file)

        kp, des = surf.detectAndCompute(frame, None)
        kmeansTrainer.add(np.float32(des))
        desc.append([kp, des])
        # train_y.append(y)

    vocabulary = kmeansTrainer.cluster()
    bow_ext.setVocabulary(vocabulary)
    print "bow vocab", np.shape(vocabulary)

    for index, file in enumerate(glob(TRAIN + "*")):
        y = int(file.split("/")[1][0])
        frame = cv2.imread(file, 0)
        histogram = bow_ext.compute(frame, desc[index][0])

        train_x.extend(histogram)
        train_y.append(y)

    print "svm items", np.shape(train_x)
    print "labels", np.shape(train_y)
    rtree_params = dict(max_depth=5, min_sample_count=5, use_surrogates=False,
                        max_categories=15, calc_var_importance=True,
                        nactive_vars=0, max_num_of_trees_in_the_forest=1000,
                        termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    rf.train(np.float32(train_x), cv2.ml.ROW_SAMPLE, np.array(train_y))

    acc = 0.
    total = 0.
    for file in glob(TRAIN + "*"):
        y = file.split("/")[1][0]
        frame = cv2.imread(file)

        kp, des = surf.detectAndCompute(frame, None)
        histogram = bow_ext.compute(frame, kp)
        value = rf.predict(histogram)

        if int(value[1][0][0]) != int(y):
            acc += 1.
        total += 1.

    print "Train Accuracy: ", (total - acc) / total

    acc = 0.
    total = 0.
    for file in glob(TEST + "*"):
        print file
        y = file.split("/")[1][0]
        frame = cv2.imread(file)
        kp, des = surf.detectAndCompute(frame, None)
        histogram = bow_ext.compute(frame, kp)
        value = rf.predict(histogram)
        cv2.drawKeypoints(frame, kp, frame)
        if int(value[1][0][0]) != int(y):
            acc += 1.
        total += 1.
        # cv2.imshow(str(value[1][0][0]) + " -- " + str(y), frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print "Test Accuracy: ", (total - acc) / total

    runCapture(rf)

if __name__ == '__main__':
    main()
