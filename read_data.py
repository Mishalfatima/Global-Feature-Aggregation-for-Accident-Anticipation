from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
from tqdm import tqdm
import cv2
import os
import tarfile
import numpy as np
import math
import sys
from PIL import Image

# from __future__ import unicode_literals
import numpy as np
import pickle as pkl
from PIL import Image
import time
import argparse

class Dashcam_data():
    def __init__(self, url='http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz',
                 dataset='HEV1', dir='./data/', batch_size=10, frame_size=[112, 112],
                 train="train", seq=False, mean_file='mean_file.npy', clip_len = 16):
        self.url = url
        self.dataset = dataset
        self.dir = dir
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.train = train
        self.mean_file = mean_file
        self.dir_structure = {}
        self.im_names = []
        self.im_pointer = 0
        self.batch = []
        self.aug_steps = [1]
        self.im_pointer = 0

        self.paths = []
        self.labels = []
        self.paths_abnormal =[]
        self.labels_abnormal = []
        self.frame_labels = []
        self.features = []
        self.feature_pointer = 0

        pickle_in = open("/hdd/local/sda/mishal/CrashCatcher/A3D_labels.pkl", "rb")
        self.labels_dict = pkl.load(pickle_in)

        if (self.train == True):

            self.cat_path_normal = "/hdd/local/sda/mishal/pyflow/ECA_Data/train/back_sub_normal"
            self.cat_path_abnormal = "/hdd/local/sda/mishal/pyflow/ECA_Data/train/back_sub_abnormal"
            self.cat_path_features = "/hdd/local/sda/mishal/pyflow/ECA_Data/features/training/"

        elif (self.train == False):
            self.cat_path_normal = "/hdd/local/sda/mishal/Anticipating-Accidents-master/dataset/videos/testing/frames/negative"
            self.cat_path_abnormal = "/hdd/local/sda/mishal/pyflow/ECA_Data/test_videos"
            self.cat_path_features = "/hdd/local/sda/mishal/pyflow/ECA_Data/features/testing/"


        for folder in os.listdir(self.cat_path_normal):

            path_normal = os.path.join(self.cat_path_normal, folder)
            self.paths.append(path_normal)
            q = self.one_hot(0)
            self.labels.append(q)
            l = np.zeros((100))

            self.frame_labels.append(l)

        for folder in os.listdir(self.cat_path_abnormal):

            path_abnormal = os.path.join(self.cat_path_abnormal, folder,"images")
            self.paths.append(path_abnormal)
            l = self.labels_dict[folder]['target']
            self.frame_labels.append(l)
            e = self.one_hot(1)
            self.labels.append(e)

        for file in os.listdir(self.cat_path_features):

            path = self.cat_path_features + file
            self.features.append(path)


        self.total_features = len(self.features)
        self.feature_ind=list(range(self.total_features))
        self.total_videos = len(self.paths)

        self.vid_ind = list(range(len(self.paths)))

    def one_hot(self,y_, n_classes=2):
        # Function to encode neural one-hot output labels from number indexes
        # e.g.:
        # one_hot(y_=[[5], [0], [3]], n_classes=6):
        #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = [int(y_)]
        # y_ = y_.reshape(len(y_))

        return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


    def get_all_im_names(self,cat_path):
        # print(self.cat_path)
        if os.path.exists('im_names.py'):
            self.im_names = np.load('im_names.npy')
        else:
            for dir, subdir, files in os.walk(cat_path):
                tif_files = [x for x in files if x.endswith('.jpg')]
                if len(tif_files) != 0:
                    seq_file_names = [dir + '/' + x for x in tif_files]
                    seq_file_names = np.array(seq_file_names)
                    # print(np.sort(seq_file_names))
                    if len(self.im_names) != 0:
                        self.im_names = np.concatenate((self.im_names, seq_file_names))
                        # self.im_names = [self.im_names, seq_file_names]
                    else:
                        self.im_names = seq_file_names
            self.im_names = np.sort(self.im_names)

        return self.im_names
        # self.im_names.sort()



    def get_mean_file(self):
        print("Computing the mean of all training frames")
        for im_count in range(len(self.im_names)):
            if im_count == 0:
                im_sum = cv2.imread(self.im_names[im_count])
            else:
                im = cv2.imread(self.im_names[im_count])
                im_sum += im
            if im_count % 100 == 0:
                print("Computed the mean of {} frames".format(im_count))
        self.mean_im = im_sum / (1.0 * len(self.im_names) * 255.0)

        # gray scale conversion
        if np.ndim(self.mean_im) > 2:
            self.mean_im = cv2.cvtColor(self.mean_im.astype(np.float32), cv2.COLOR_RGB2GRAY)

        np.save(self.mean_file, self.mean_im)

    def get_next_batch(self):

        if (self.im_pointer == 0):
            np.random.shuffle(self.vid_ind)

        self.l = []
        self.batch = []
        self.fl = []

        for idx in range(self.batch_size):
            self.batch.append(self.paths[self.vid_ind[self.im_pointer]])
            self.l.append(self.labels[self.vid_ind[self.im_pointer]])
            self.fl.append(self.frame_labels[self.vid_ind[self.im_pointer]])

            self.im_pointer += 1
            if (self.im_pointer == len(self.paths)):
                self.im_pointer = 0
                np.random.shuffle(self.vid_ind)

        self.batch = np.array(self.batch)
        self.l = np.array(self.l)


        return self.batch, self.l, self.fl


    def get_next_batch_features(self):

        if (self.feature_pointer == 0):
            np.random.shuffle(self.feature_ind)

        self.batch = np.zeros((self.batch_size,100,20,4096))
        self.labels = np.zeros((self.batch_size,2))
        self.start = np.zeros((self.batch_size))
        self.paths = []
        self.end = np.zeros((self.batch_size,100))

        for idx in range(self.batch_size):
            ww = np.ones((100))
            #arr = np.zeros((100))
            f = self.features[self.feature_ind[self.feature_pointer]]

            batch = np.load(f)
            self.batch[idx,:,:,:] = batch['data']
            self.labels[idx,:] = batch["labels"][0]
            self.paths.append(batch['paths'][0])

            if self.labels[idx,:][0] == 0:
                video_name = batch['paths'][0]
                video_name = video_name.split("/")
                video_name = video_name[-2]
                array = self.labels_dict[video_name]['target']

                if (len(array)>100):
                    arr = array[len(array)-100: len(array)]
                    arr1 = array[0:100]

                    itemindex = np.argwhere(arr == 1)
                    itemindex1 = np.argwhere(arr1 == 1)

                    if (list(itemindex1)!=[]):
                        s = itemindex1.min()
                    else:
                        s = itemindex.min()


                else:
                    arr = array

                    '''index = len(array)

                    arr[0:index] = array'''

                    itemindex = np.argwhere(arr == 1)
                    s = itemindex.min()

                    zeros = 100-len(array)
                    ii = np.zeros((zeros))
                    ww[len(array):100]= ii

                self.start[idx] = s

            else:
                self.start[idx] = 0

            self.end[idx,:] = ww

            self.feature_pointer += 1
            if (self.feature_pointer == len(self.features)):
                self.feature_pointer = 0
                np.random.shuffle(self.feature_ind)


        '''print(self.batch.shape)
        print(self.labels.shape)
        print(self.paths)
        print(self.start)
        print(self.end)'''
        return self.batch, self.labels,self.paths,self.start,self.end




if __name__ == '__main__':

    dataset = Dashcam_data(train=False)

    im_names = (dataset.total_features)
    tot_batches = int(im_names/10)
    acc = 0
    accg=0
    for i in range(tot_batches):
        print(i, "out of", tot_batches)
        batch,labels, paths, start,end = dataset.get_next_batch_features()


        ''''for j in range(10):
            if labels[j][0]==0:
                if (start[j] == 0):
                    print(paths[j])
                    print("HAHAHAH")
                    break
                accg=accg+start[j]
                acc+=1'''

    '''print(acc)
    print(accg)'''















