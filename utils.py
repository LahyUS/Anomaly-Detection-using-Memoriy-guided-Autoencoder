import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from matplotlib.pyplot import figure
from matplotlib import colors

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list))
        anomaly_score_list.append(score)
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def plot_ROC(anomal_scores, labels, auc, log_dir, dataset_type, method, trained_model_using):
    # plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=np.squeeze(
        labels, axis=0), y_score=np.squeeze(anomal_scores))

    # create ROC curve
    plt.title('Receiver Operating Characteristic \nmethod: ' +
              method + ', dataset: ' + dataset_type +
              ', trained model used: ' + trained_model_using)
    plt.plot(fpr, tpr, 'b', label='ROC curve (AUC = %0.4f)' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', label='random predict')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    #plt.plot([0, 1], [1, 0], color='black', linewidth=1.5, linestyle='dashed')
    #plt.legend(loc='lower right')

    plt.savefig(os.path.join(log_dir, 'ROC.png'))

def AUC(anomal_scores, labels):
    # y_test = y_true=np.squeeze(labels, axis=0)
    # y_pred_proba = y_score=np.squeeze(anomal_scores)
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))

    return frame_auc

def plot_anomaly_scores(anomaly_score_total_list, labels, log_dir, dataset_type, method, trained_model_using):
    matrix = np.array([labels == 1])

    # Mask the False occurences in the numpy array as 'bad' data
    matrix = np.ma.masked_where(matrix == True, matrix)

    # Create a ListedColormap with only the color green specified
    cmap = colors.ListedColormap(['none'])

    # Use the `set_bad` property of `colormaps` to set all the 'bad' data to red
    cmap.set_bad(color='lavenderblush')
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    plt.title('Anomaly score/frame, method: ' +
              method + ', dataset: ' + dataset_type +
              ', trained model used: ' + trained_model_using)
    ax.pcolormesh(matrix, cmap=cmap, edgecolor='none', linestyle='-', lw=1)

    y = anomaly_score_total_list
    x = np.arange(0, len(y))
    plt.plot(x, y, color="steelblue", label="score/frame")
    plt.legend(loc='lower left')
    plt.ylabel('Score')
    plt.xlabel('Frames')
    plt.savefig(os.path.join(log_dir, 'anomaly_score.png'))

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def optimalThreshold(anomal_scores, labels):
    y_true = 1 - labels[0, :1962]
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
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.scatter(fpr[ix], tpr[ix],  marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #pyplot.legend()
    # show the plot
    #pyplot.show()
    #return threshold[ix]
    #anomaly_score_total_list, np.expand_dims(1-labels_list, 0)
    #print()
    return threshold[ix]
