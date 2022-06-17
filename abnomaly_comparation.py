import argparse
import cv2 as cv
from Image_Similarity import CompareHistogram as ch
from Image_Similarity import CompareFeatures as cf
from Image_Similarity import Image_Diff as id
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
from matplotlib import pyplot
# Import module for data visualization


def optimalThreshold(anomal_scores, labels):
    y_true = 1 - labels[0, 4:1966]
    y_true  = np.squeeze(y_true)
    y_score = np.squeeze(anomal_scores[:1962])
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    frame_auc = roc_auc_score(y_true, y_score)
    print("AUC: {}".format(frame_auc))
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix], tpr[ix],  marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    #pyplot.legend()
    # show the plot
    #pyplot.show()
    #return threshold[ix]
    #anomaly_score_total_list, np.expand_dims(1-labels_list, 0)
    #print()
    return threshold[ix]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", type=str, default='./datasets/testing/frames/01/005.jpg', help="first input image")
    ap.add_argument("-s", "--second", type=str, default='./datasets/predicted/pred/frames/0001.jpg', help="second")
    ap.add_argument("-m", "--mode", type=int, default=2, help="Mode")
    args = vars(ap.parse_args())

    time_t = 0
    test_input_path = []
    for i in range(178):
        frame_th = ''
        frame_num = time_t + i
        if frame_num < 10:
            frame_th = '00' + str(frame_num)
        elif frame_num < 100:
            frame_th = '0' + str(frame_num)
        else:
            frame_th = str(frame_num)
        test_input_path.append("./datasets/testing/frames/01/" + frame_th + ".jpg")

    pred_input_path = []
    for i in range(174):
        frame_th = ''
        frame_num = time_t + i
        if frame_num < 10:
            frame_th = '00' + str(frame_num)
        elif frame_num < 100:
            frame_th = '0' + str(frame_num)
        else:
            frame_th = str(frame_num)
        pred_input_path.append("./datasets/predicted/pred/frames/0" + frame_th + ".jpg")

    test_input_imgs = []
    for i in range(178):
        img = cv.imread(test_input_path[i])
        test_input_imgs.append(img)

    pred_input_imgs = []
    for i in range(174):
        img = cv.imread(pred_input_path[i])
        pred_input_imgs.append(img)
    #t_minus_one_frames = input_imgs[:4]
    #tensor_imgs = torch.tensor(t_minus_one_frames)

    # load frame score
    frame_scores = np.load("./datasets/predicted/anomaly_score.npy")
    labels = np.load('./data/frame_labels_'+'ped2'+'.npy')

    opt_threshold = optimalThreshold(frame_scores, labels)

    test_img = test_input_imgs[1]
    for i in range(174):
        # load the two input images
        imageA = test_input_imgs[i+4]  # test frame
        imageB = pred_input_imgs[i]  # pred frame

        #imageA = cv.imread(args["first"])  # test frame
        #imageB = cv.imread(args["second"])  # pred frame

        # load mode of program
        mode = int(args["mode"])

        # resize image
        w1, h1, c1 = imageA.shape
        w2, h2, c2 = imageB.shape

        if w1 != 256:
            imageA = cv.resize(imageA, (256, 256))
        if w2 != 256:
            imageB = cv.resize(imageB, (256, 256))

        # Show images
        #cv.imshow("Test frame", cv .resize(imageA, None, fx=1, fy=1))
        #cv.imshow("Pred frame", cv.resize(imageB, None, fx=1, fy=1))

        # Run compare modules
        if mode == 0:
            ch.compareHistogram(imageA, imageB)
        elif mode == 1:
            cf.compareSIFT(imageA, imageB)
        elif mode == 2:
            id.image_differences(imageA, imageB, frame_scores[i], opt_threshold)
        #cv.waitKey(0)
        cv.imshow("Test frame compared", cv.resize(imageA, None, fx=1, fy=1))
        #cv2.imwrite("Result_Original.png", imageA)
        cv.imshow("Pred frame compared", cv.resize(imageB, None, fx=1, fy=1))
        cv.waitKey(5)
        # Press Q on keyboard to stop recording
        if cv.waitKey(1) & 0xFF == 27:
                break

        # Closes all the frames
    cv.destroyAllWindows()