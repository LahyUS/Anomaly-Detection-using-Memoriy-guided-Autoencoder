import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from utils import *
import glob
import time
import argparse
import time
from datetime import datetime

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size for training')
parser.add_argument('--test_batch_size', type=int,
                    default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256,
                    help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred',
                    help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512,
                    help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512,
                    help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10,
                    help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01,
                    help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1,
                    help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='avenue',
                    help='type of dataset: ped1, ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str,
                    default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str,
                    default='./my_trained_model/2/avenue_prediction_model.pth', help='directory of model')
parser.add_argument('--m_items_dir', type=str,
                    default='./my_trained_model/2/avenue_prediction_keys.pt', help='directory of model')
parser.add_argument('--exp_dir', type=str, default='log',
                    help='directory of log')

start_time = datetime.now()
print("Start time:", start_time.strftime("%d/%m/%Y %H:%M:%S"))

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

# make sure to use cudnn for computational performance
torch.backends.cudnn.enabled = True

test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# load the dataset and convert a ndarray image/frame into a float tensor. Then scale the image/frame
# pixel intensity value in the range [-1, 1]
test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor(), ]),
                          resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
# dataset length
test_size = len(test_dataset)

# load data into mini test batch with batch_size:
#   + test_dataset: loader dataset
#   + batch_size: size of mini batch
#   + shuffle: not shuffle due to sequential data
#   + num_workers: how many subprocesses to use for data loading
#   + drop_last: If the size of dataset is not divisible by the batch size, then the last batch will be smaller
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# define Mean Error Loss
loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
# load labels file of dataset
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

# Setup a list contain video segments, element in the list contain all frames of this video segment.
videos = OrderedDict()  # './dataset/ped2/testing/frames/01'; .../02; ...
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

# initialize label list, psnr dict and feature_distance_list (compactness loss) dict
labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Start Evaluation of:', args.dataset_type + ',', 'method:', args.method)

# setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        from_frame = 4+label_length
        to_frame = videos[video_name]['length']+label_length
        frame_labels = labels[0][from_frame:to_frame]
        labels_list = np.append(labels_list, frame_labels)
    else:
        from_frame = label_length
        to_frame = videos[video_name]['length']+label_length
        frame_labels = labels[0][from_frame:to_frame]
        labels_list = np.append(labels_list, frame_labels)

    # update indices
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

# Iterate on each frame of the whole dataset, forward through the model
# predict: img ndim = 4, shape ([1, 15, 256, 256])
# recons: img ndim = 4, shape ([1, 3, 256, 256])
for k, (imgs) in enumerate(test_batch):

    # imgSrc_clone = torch.clone(imgs)
    # imgSrc_clone = imgSrc_clone[0].permute(1, 2, 0)
    # imgSrc_clone = imgSrc_clone.cpu().detach().numpy()

    # imgSrc_clone = (imgSrc_clone + 1) * 127.5  # revert range
    # imgSrc_clone = imgSrc_clone.astype(dtype=np.uint8)

    # cv2.imshow('image window', imgSrc_clone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num]
                                   .split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num]
                                   .split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(
            imgs[:, 0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0, 3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:, 3*4:])

    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(
            imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse(
            (outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # imgRecon_clone = torch.clone(imgs)
        # imgRecon_clone = imgRecon_clone[0].permute(1, 2, 0)
        # imgRecon_clone = imgRecon_clone.cpu().detach().numpy()

        # imgRecon_clone = (imgRecon_clone + 1) * 127.5  # revert range
        # imgRecon_clone = imgRecon_clone.astype(dtype=np.uint8)

        # cv2.imshow('image window', imgRecon_clone)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    if point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    # calculate psnr for each frame and then append it to psnr list
    psnr_score = psnr(mse_imgs)
    psnr_index = videos_list[video_num].split('/')[-1]
    psnr_list[psnr_index].append(psnr_score)
    # append compactness lost of current frame to compactness list
    feature_distance_list[videos_list[video_num].split(
        '/')[-1]].append(mse_feas)


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]

    psnr_list_of_video = psnr_list[video_name]
    # min-max normalization for PSNR
    anomaly_score_list_of_video = anomaly_score_list(psnr_list_of_video)

    feature_distance_list_of_video = feature_distance_list[video_name]
    # min-max normalization for compactness loss
    anomaly_score_list_inv_of_video = anomaly_score_list_inv(
        feature_distance_list_of_video)

    # Sum score for anomaly rate
    score = score_sum(anomaly_score_list_of_video,
                      anomaly_score_list_inv_of_video, args.alpha)

    # Append score to total list
    anomaly_score_total_list += score


anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

plotROC(anomaly_score_total_list, np.expand_dims(
    1-labels_list, 0), accuracy, log_dir)

print('The result of', args.dataset_type)
print('AUC:', accuracy*100, '%')


end_time = datetime.now()
time_range = end_time-start_time
print('Evaluate is taken: ', time_range)
