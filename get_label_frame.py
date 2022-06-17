import numpy as np
import h5py
from scipy.io import loadmat
data = loadmat(
    r"/home/huu/thesis/anodetection-aemem/dataset/ped1/ped1.mat")
arr = np.zeros(0, np.int64)

for rangeAno in data['gt'][0]:
    start = rangeAno[0][0]
    end = rangeAno[1][0]
    arrItem = np.zeros(200, np.int64)
    for index in range(200):
        if index > start and index < end:
            arrItem[index] = 1
        else:
            arrItem[index] = 0
    arr = np.concatenate((arr, arrItem), axis=None)
    print('Start: ', rangeAno[0][0])
    print('End: ', rangeAno[1][0], '\n')
print('Done')
arr = np.array([arr])
np.save('frame_labels_ped1.npy', arr)
