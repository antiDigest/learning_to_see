

import numpy as np
import cv2
from glob import glob
from utils import *


TRAIN = "data/train/"
TEST = "data/test/"

surf = cv2.xfeatures2d.SURF_create(extended=True)
bow_ext = cv2.BOWImgDescriptorExtractor(
    surf, cv2.BFMatcher(cv2.NORM_L2))
svm = cv2.ml.SVM_create()
rf = cv2.ml.RTrees_create()
images = ['data/find1(1).jpg', 'data/find1(2).jpeg',
          'data/find1(3).jpg', 'data/find1(5).JPG']
imgs = ['data/find1(7).jpg', 'data/find1(6).jpg', 'data/find1(8).jpeg']
imgsWow = ['data/find1(9).jpg', 'data/find1(10).jpg', 'data/find1(11).jpeg']


def extract_histogram(window):
    kp, des = surf.detectAndCompute(window, None)
    histogram = bow_ext.compute(window, kp)
    return histogram


def detect_logo(image, clf):
    maxVote = 0
    found = None
    voteList = []

    for resized in pyramid(image):
        for (winW, winH) in windows:
            for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):

                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                histogram = extract_histogram(window)
                value = clf.predict(histogram)

                if value[0] == 1.0:
                    votes = svm.getVotes(histogram, 0).astype('float')
                    votes = (votes[1, :]) / sum(votes[1, :])

                    if (votes[0] - 0.2) > (votes[1]):
                        voteList.append((votes[0], (x, y, x + winW, y + winH),
                                         frame.shape[1] / resized.shape[1]))
                        # if (votes[0] > maxVote):
                        #     maxVote = votes[0]
                        #     found = (maxVote, (x, y, x + winW, y + winH),
                        #              frame.shape[1] / resized.shape[1])

    if voteList != []:
        voteList = sorted(voteList, key=lambda x: x[0], reverse=True)[:15]
        boxes = np.array([box for (vote, box, ratio) in voteList])
        ratios = np.array([ratio for (vote, box, ratio) in voteList])
        boxes, ratios = non_max_suppression(boxes, 0.65, ratios=ratios)

        clone = frame.copy()
        for found in zip(boxes, ratios):
            (window, r) = found
            startx, starty, endx, endy = (int(w) * r for w in window)
            cv2.putText(clone, "Logo", (startx, starty),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))
            cv2.rectangle(clone, (startx, starty),
                          (endx, endy), (0, 255, 0), 2)

    return clone


def video_capture(clf):
    cap = cv2.VideoCapture(0)

    # for image in imgsWow:
    while(1):
        ret, frame = cap.read()
        # print image

        # frame = cv2.imread(image)

        frame = detect_logo(frame, clf)

        cv2.imshow("UTD LOGO", frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # break


def collect_vocab(kmeansTrainer):
    desc = []

    for file in glob(TRAIN + "*"):
        frame = cv2.imread(file)

        kp, des = surf.detectAndCompute(frame, None)
        kmeansTrainer.add(np.float32(des))
        desc.append([kp, des])

    return desc, kmeansTrainer


def collect_train(desc):

    train_y = []
    train_x = []

    for index, file in enumerate(glob(TRAIN + "*")):
        y = int(file.split("/")[2][0])
        frame = cv2.imread(file, 0)
        # print index
        histogram = bow_ext.compute(frame, desc[index][0])

        train_x.extend(histogram)
        train_y.append(y)

    return train_x, train_y


def calc_accuracy(FOLDER, clf):
    print FOLDER
    acc = 0.
    total = 0.
    for file in glob(FOLDER + "*"):
        y = file.split("/")[2][0]
        frame = cv2.imread(file)

        kp, des = surf.detectAndCompute(frame, None)
        histogram = bow_ext.compute(frame, kp)
        value = clf.predict(histogram)

        if value is not None:
            if int(value[0]) != int(y):
                acc += 1.
            total += 1.

    print "Accuracy: ", (total - acc) / total


def train(clf):

    kmeansTrainer = cv2.BOWKMeansTrainer(64)
    print "Collecting Bag-Of-Words for dataset"
    desc, kmeansTrainer = collect_vocab(kmeansTrainer)

    print "Creating Bag-Of-Words Vocabulary"
    vocabulary = kmeansTrainer.cluster()
    bow_ext.setVocabulary(vocabulary)

    print "Collecting training items"
    train_x, train_y = collect_train(desc)

    # rtree_params = dict(max_depth=5, min_sample_count=5, use_surrogates=False,
    #                     max_categories=15, calc_var_importance=True,
    #                     nactive_vars=0, max_num_of_trees_in_the_forest=1000,
    #                     termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    clf.train(np.float32(train_x), cv2.ml.ROW_SAMPLE, np.array(train_y))

    print "Calculating accuracy:"
    calc_accuracy(TRAIN, clf)
    calc_accuracy(TEST, clf)

    return clf

if __name__ == '__main__':
    clf = train(svm)
    video_capture(clf)
