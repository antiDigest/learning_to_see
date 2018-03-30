import numpy as np
import cv2
from glob import glob

import boxes


TRAIN = "train/"
TEST = "test/"

surf = cv2.xfeatures2d.SURF_create(extended=True)
bow_ext = cv2.BOWImgDescriptorExtractor(
    surf, cv2.BFMatcher(cv2.NORM_L2))
kmeansTrainer = cv2.BOWKMeansTrainer(80)
svm = cv2.ml.SVM_create()
rf = cv2.ml.RTrees_create()

box1 = {'width': 200, 'height': 200}
box2 = {'width': 200, 'height': 400}
box3 = {'width': 400, 'height': 200}

boxlist = [box1, box2, box3]


def drawRectangle(frame, x, y, width, height):

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

    return frame


def checkBox(frame, width, height):

    w, h, _ = frame.shape

    boxPreds = []
    for x in range(1, w - width):
        for y in range(1, h - height):
            boxPred = {}
            im = frame[x:x + width, y:y + height]
            kp, des = surf.detectAndCompute(im, None)
            histogram = bow_ext.compute(im, kp)
            boxPred['value'] = rf.getVotes(histogram)
            print boxPred['value']
            cv2.drawKeypoints(frame, kp, boxPred['frame'])
            boxPred['box'] = [x, y, width, height]
            boxPreds.append(boxPred)

    boxPred = sorted(boxPreds, lambda x: x['value'])[0]

    cv2.putText(boxPred['frame'], string, (50, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    return boxPred


def runCapture(clf):
    cap = cv2.VideoCapture(0)

    while(1):
        ret, frame = cap.read()

        boxPreds = []
        for box in boxlist:
            boxPreds.append(checkBox(frame, box['width'], box['height']))

        boxPred = sorted(boxPreds, lambda x: x['value'])[0]

        x, y, width, height = boxPred['box']

        im = drawRectangle(frame, x, y, width, height)

        cv2.imshow("Temoc", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():

    train_y = []
    train_x = []

    desc = []

    for file in glob(TRAIN + "*"):
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
                        max_categories=15, calc_var_importance=False,
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
