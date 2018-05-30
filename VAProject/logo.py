

"""
    @author: Antriksh Agarwal
    Version 0: 04/29/2018
"""

import numpy as np
import cv2
from glob import glob
from utils import *
import os
import pickle


TRAIN = "data/train/"
TEST = "data/test/"

surf = cv2.xfeatures2d.SURF_create(extended=True)
bow_ext = cv2.BOWImgDescriptorExtractor(
    surf, cv2.BFMatcher(cv2.NORM_L2))
kmeansTrainer = cv2.BOWKMeansTrainer(64)
# clf = cv2.ml.SVM_create()
clf = cv2.ml.RTrees_create()
images = ['data/find1(1).jpg', 'data/find1(2).jpeg',
          'data/find1(3).jpg', 'data/find1(5).JPG']
imgs = ['data/find1(7).jpg', 'data/find1(6).jpg', 'data/find1(8).jpeg']
imgsWow = ['data/find1(9).jpg', 'data/find1(10).jpg', 'data/find1(11).jpeg']


def init():
    try:
        if os.path.exists('models/random_forest.xml') and os.path.exists('models/bow_pickle.pickle'):
            with open('models/bow_pickle.pickle', 'r') as f:
                vocab = pickle.load(f)
                bow_ext.setVocabulary(vocab)
            clf.load('models/random_forest.xml')
            calc_accuracy(TEST)
        else:
            train()
    except:
        train()


def detect_logo(image):
    kp, des = surf.detectAndCompute(image, None)
    histogram = bow_ext.compute(image, kp)
    value = clf.predict(histogram)

    if value[0] == 1.0:
        return True
    return False


def video_capture():
    cap = cv2.VideoCapture(0)

    for image in imgsWow:
        # while(1):
        # ret, frame = cap.read()
        print image

        frame = cv2.imread(image)

        frame = detect_logo(frame)

        cv2.imshow("UTD LOGO", frame)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # break


def collect_vocab():
    desc = []

    for file in glob(TRAIN + "*"):
        frame = cv2.imread(file)

        kp, des = surf.detectAndCompute(frame, None)
        kmeansTrainer.add(np.float32(des))
        desc.append([kp, des])

    print "Creating Bag-Of-Words Vocabulary"
    vocabulary = kmeansTrainer.cluster()
    bow_ext.setVocabulary(vocabulary)
    # bow_ext.save('models/bow_ext.xml')

    with open('models/bow_pickle.pickle', 'w') as f:
        pickle.dump(vocabulary, f)

    return desc, vocabulary


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


def calc_accuracy(FOLDER):
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


def train():

    print "Collecting Bag-Of-Words for dataset"
    desc, vocabulary = collect_vocab()

    print "Collecting training items"
    train_x, train_y = collect_train(desc)

    # print vocabulary.shape
    # print len(train_x)
    # print len(train_y)

    # rtree_params = dict(max_depth=5, min_sample_count=5, use_surrogates=False,
    #                     max_categories=15, calc_var_importance=True,
    #                     nactive_vars=0, max_num_of_trees_in_the_forest=1000,
    #                     termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    clf.train(np.float32(train_x), cv2.ml.ROW_SAMPLE, np.array(train_y))
    print "Saving Model"
    clf.save('models/random_forest.xml')

    print "Calculating accuracy:"
    calc_accuracy(TRAIN)
    calc_accuracy(TEST)

if __name__ == '__main__':
    if os.path.exists('models/random_forest.xml') and os.path.exists('models/bow_pickle.pickle'):
        with open('models/bow_pickle.pickle', 'r') as f:
            bow_ext.setVocabulary(pickle.load(f))
        clf.load('models/random_forest.xml')
    else:
        train()
    video_capture()

    img1 = cv2.imread('box.png', 0)          # queryImage
    img2 = cv2.imread('box_in_scene.png', 0)  # trainImage
