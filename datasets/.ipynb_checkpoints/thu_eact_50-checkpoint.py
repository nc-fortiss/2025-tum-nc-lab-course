import numpy as np
import os
import re
from os import listdir
from os.path import join
from scipy import io
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from .process import *

repr_map = {'eventFrame':get_eventFrame,
            'eventAccuFrame':get_eventAccuFrame,
            'timeSurface':get_timeSurface,
            'eventCount':get_eventCount}

#known_actions = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 
#                 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49]
#unknown_actions = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]

known_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
unknown_actions = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

# left or right move all event locations randomly
def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

# flip half of the event images along the x dimension
def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

# randomly drop events
def random_drop_events(events, drop_factor=0.2):
    events = events
    return events

# randomly zoom in  or out
def random_zoom_events(events, max_zoom=0.5, resolution=(180, 240)):
    H, W = resolution
    zoom = (1-max_zoom) + 2 * max_zoom * np.random.rand()
    events[:,0] = events[:,0] * zoom
    events[:,1] = events[:,1] * zoom

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events



class THU_EACT_50(Dataset):
    def __init__(self, path="../THU_EACT_50", mode="front", train=True, augmentation=False, max_points=1000000,
                 repr=['timeSurface'], sampling_time=10, sample_length=1000, ds_factor=3, center_crop=True):
        super(THU_EACT_50, self).__init__()
        list_file_name = None
        eval = not train
        if mode == "front": # front views (C1-C2)
            list_file_name = join(path,"test.txt") if eval else join(path,"train.txt")
            valid_labels = known_actions + unknown_actions
        elif mode == "pretrain": # subset
            #list_file_name = join(path,"known_test_new.txt") if eval else join(path,"known_train_new.txt")
            list_file_name = join(path,"test.txt") if eval else join(path,"train.txt")
            valid_labels = known_actions
        elif mode == "clp": # subset
            #list_file_name = join(path,"unknown_test_new.txt") if eval else join(path,"unknown_train_new.txt")
            list_file_name = join(path,"test.txt") if eval else join(path,"train.txt")
            valid_labels = unknown_actions

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.max_points = max_points
        self.datafile = path
        self.train = train

        self.repr = repr
        self.time_num = sample_length // sampling_time
        self.sample_length = sample_length * 1000 # to us
        self.sampling_time = sampling_time * 1000 # to us

        self.ds_factor = ds_factor
        self.center_crop = center_crop

        #known_file = open(known_file_name, "w")
        #unknown_file = open(unknown_file_name, "w")
        list_file = open(list_file_name, "r")
        for line in list_file:
            file, label = line.split(",")
            if int(label) in valid_labels:
                self.files.append(file)
                self.labels.append(int(label))
            #if int(label) in known_actions:
            #    known_file.write(file + ',' + label)
            #elif int(label) in unknown_actions:
            #    unknown_file.write(file + ',' + label)
            
        list_file.close()
        #known_file.close()
        #print('Completed write ', known_file_name)
        #unknown_file.close()
        #print('Completed write ', unknown_file_name)

        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = os.path.join(self.datafile, self.files[idx])


        # read the raw csv data and calculate the representations
        #pd_reader = pd.read_csv(f, header=None).values
        #events = np.vstack((pd_reader[:, 1], pd_reader[:, 0], pd_reader[:, 4], pd_reader[:, 3])).T.astype(np.float32)
        # read npy data and calculate the representations
        f = f.replace("csv", "npy")
        events = np.load(f)

        events = events[events[:,3]!=0.] # delete all the points that have the polarity of 0

        # normalize the timestamps
        _min = events[:,2].min()
        _max = events[:,2].max()
        events[:,2] = (events[:,2] - _min) #/ (_max - _min)

        # cut to sample_length
        if self.augmentation: #self.train:
            t_start = np.random.randint(max(int(events[:,2].max()) - self.sample_length, 1,))
            #print(t_start)
            #print(t_start+self.sample_length)
        else:
            t_start = 0
        events = events[events[:, 2] >= t_start]
        events = events[events[:, 2] < (t_start+self.sample_length)]
        events[:,2] = (events[:,2] - t_start)

        #print(events.shape)

        if self.augmentation:
            events = random_shift_events(events, max_shift=50, resolution=(800, 1200))
            events = random_flip_events_along_x(events, resolution=(800, 1200))
            events = random_zoom_events(events, max_zoom=0.2, resolution=(800, 1200))

        # center crop
        if self.center_crop:
            events = events[events[:, 0] >= 300]
            events = events[events[:, 0] < 900]
            events[:, 0] = events[:, 0] - 300
            events = events[events[:, 1] >= 100]
            events = events[events[:, 1] < 700]
            events[:, 1] = events[:, 1] - 100

        # flip in y
        events[:, 1] = 600 - 1 - events[:, 1]

        # downsize
        events[:, 0] = events[:, 0] / self.ds_factor
        events[:, 1] = events[:, 1] / self.ds_factor

        if self.repr == ['myCount']:
            reprs = np.zeros((2, self.time_num, 600//self.ds_factor, 600//self.ds_factor))
            events[:, 2] = events[:, 2] / self.sampling_time
            events[:, 3] = (events[:, 3] + 1.0) / 2.0
            np.add.at(reprs, (np.floor(events[:, 3]).astype(np.int32), np.clip(np.floor(events[:, 2]).astype(np.int32), 0, self.time_num-1), 
                              np.floor(events[:, 1]).astype(np.int32), np.floor(events[:, 0]).astype(np.int32)), 1)
        else:
            reprs = []
            for repr_name in self.repr:
                repr_array = repr_map[repr_name](events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32),
                                                events[:, 3], repr_size=(200, 200), time_num=self.time_num)
            # standardization
            # mu = np.mean(repr_array)
            # sigma = np.std(repr_array)
            # repr_array = (repr_array - mu) / sigma

            # show
            #anim = show_repr(repr_array)
            #anim.save('anims/animation_' + str(idx) + '_label' + str(label) + '_' + repr_name + '.mp4', fps=100)

                reprs.append(repr_array)
            reprs = np.array(reprs)
        reprs = torch.tensor(reprs, dtype=torch.float)
        #print(torch.max(reprs))
        # time to last dim
        reprs = reprs.permute(0, 2, 3, 1)
        # experimental thresholding
        #threshold = 5
        #reprs = reprs * (reprs > threshold)
        #reprs = reprs - threshold
        return reprs, label
    
    def save_to_disk_npy(self, new_path=None):
        """
        saves events in npy format
        """
        if new_path:
            data_path = new_path
        else:
            data_path = self.datafile
        for f in self.files:
            f_csv = os.path.join(self.datafile, f)
            pd_reader = pd.read_csv(f_csv, header=None).values
            events = np.vstack((pd_reader[:, 1], pd_reader[:, 0], pd_reader[:, 4], pd_reader[:, 3])).T.astype(np.float32)
            f_csv_new = os.path.join(data_path, f)
            f_npy = f_csv_new.replace("csv", "npy")
            np.save(f_npy, events)
            print(f_npy)


class SimCLR_THU_EACT_50(Dataset):
    def __init__(self, path="../THU_EACT_50", mode="front", train=True, augmentation=False, max_points=1000000,
                 repr=['timeSurface'], sampling_time=10, sample_length=1000, ds_factor=3, n_views=2, center_crop=True):
        super(SimCLR_THU_EACT_50, self).__init__()
        list_file_name = None
        eval = not train
        if mode == "front":  # front views (C1-C2)
            list_file_name = join(path, "test.txt") if eval else join(path, "train.txt")
            # known_file_name = join(path,"known_test_new.txt") if eval else join(path,"known_train_new.txt")
            # unknown_file_name = join(path,"unknown_test_new.txt") if eval else join(path,"unknown_train_new.txt")
            valid_labels = known_actions + unknown_actions
        elif mode.startswith("view_"):  # just a single view
            list_file_name = join(path, "test_" + mode + ".txt") if eval else join(path, "train_" + mode + ".txt")
            valid_labels = known_actions + unknown_actions
        elif mode == "small":  # subset
            list_file_name = join(path, "test_small.txt") if eval else join(path, "train_small.txt")
            # known_file_name = join(path,"known4_test_small.txt") if eval else join(path,"known4_train_small.txt")
            # unknown_file_name = join(path,"unknown4_test_small.txt") if eval else join(path,"unknown4_train_small.txt")
            valid_labels = known_actions + unknown_actions
        elif mode == "mini":  # subset
            list_file_name = join(path, "test_mini.txt") if eval else join(path, "train_mini.txt")
            valid_labels = known_actions + unknown_actions
        elif mode == "small_pretrain":  # subset
            # list_file_name = join(path,"known4_test_small.txt") if eval else join(path,"known4_train_small.txt")
            list_file_name = join(path, "test_small.txt") if eval else join(path, "train_small.txt")
            valid_labels = known_actions
        elif mode == "pretrain":  # subset
            # list_file_name = join(path,"known_test_new.txt") if eval else join(path,"known_train_new.txt")
            list_file_name = join(path, "test.txt") if eval else join(path, "train.txt")
            valid_labels = known_actions
        elif mode == "small_clp":  # subset
            # list_file_name = join(path,"unknown4_test_small.txt") if eval else join(path,"unknown4_train_small.txt")
            list_file_name = join(path, "test_small.txt") if eval else join(path, "train_small.txt")
            valid_labels = unknown_actions
        elif mode == "clp":  # subset
            # list_file_name = join(path,"unknown_test_new.txt") if eval else join(path,"unknown_train_new.txt")
            list_file_name = join(path, "test.txt") if eval else join(path, "train.txt")
            valid_labels = unknown_actions

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.max_points = max_points
        self.datafile = path
        self.train = train

        self.repr = repr
        self.time_num = sample_length // sampling_time
        self.sample_length = sample_length * 1000  # to us
        self.sampling_time = sampling_time * 1000  # to us

        self.ds_factor = ds_factor
        self.center_crop = center_crop
        self.n_views = n_views

        # known_file = open(known_file_name, "w")
        # unknown_file = open(unknown_file_name, "w")
        list_file = open(list_file_name, "r")
        for line in list_file:
            file, label = line.split(",")
            if int(label) in valid_labels:
                self.files.append(file)
                self.labels.append(int(label))
            # if int(label) in known_actions:
            #    known_file.write(file + ',' + label)
            # elif int(label) in unknown_actions:
            #    unknown_file.write(file + ',' + label)

        list_file.close()
        # known_file.close()
        # print('Completed write ', known_file_name)
        # unknown_file.close()
        # print('Completed write ', unknown_file_name)

        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = os.path.join(self.datafile, self.files[idx])

        # read the raw csv data and calculate the representations
        # pd_reader = pd.read_csv(f, header=None).values
        # events = np.vstack((pd_reader[:, 1], pd_reader[:, 0], pd_reader[:, 4], pd_reader[:, 3])).T.astype(np.float32)
        # read npy data and calculate the representations
        f = f.replace("csv", "npy")
        events = np.load(f)

        events = events[events[:, 3] != 0.]  # delete all the points that have the polarity of 0

        # normalize the timestamps
        _min = events[:, 2].min()
        _max = events[:, 2].max()
        events[:, 2] = (events[:, 2] - _min)  # / (_max - _min)


        reprs_views = []
        labels = []
        for v in range(self.n_views):
            events_v = np.copy(events)
            # cut to sample_length
            if self.augmentation:  # self.train:
                t_start = np.random.randint(max(int(events_v[:, 2].max()) - self.sample_length, 1, ))
                # print(t_start)
                # print(t_start+self.sample_length)
            else:
                t_start = 0
            events_v = events_v[events_v[:, 2] >= t_start]
            events_v = events_v[events_v[:, 2] < (t_start + self.sample_length)]
            events_v[:, 2] = (events_v[:, 2] - t_start)

            # print(events.shape)

            if self.augmentation:
                events_v = random_shift_events(events_v, max_shift=50, resolution=(800, 1200))
                events_v = random_flip_events_along_x(events_v, resolution=(800, 1200))
                events_v = random_zoom_events(events_v, max_zoom=0.2, resolution=(800, 1200))

            # center crop
            if self.center_crop:
                events_v = events_v[events_v[:, 0] >= 300]
                events_v = events_v[events_v[:, 0] < 900]
                events_v[:, 0] = events_v[:, 0] - 300
                events_v = events_v[events_v[:, 1] >= 100]
                events_v = events_v[events_v[:, 1] < 700]
                events_v[:, 1] = events_v[:, 1] - 100

            # flip in y
            events_v[:, 1] = 600 - 1 - events_v[:, 1]

            # downsize
            events_v[:, 0] = events_v[:, 0] / self.ds_factor
            events_v[:, 1] = events_v[:, 1] / self.ds_factor

            if self.repr == ['myCount']:
                reprs = np.zeros((2, self.time_num, 600 // self.ds_factor, 600 // self.ds_factor))
                events_v[:, 2] = events_v[:, 2] / self.sampling_time
                events_v[:, 3] = (events_v[:, 3] + 1.0) / 2.0
                np.add.at(reprs, (np.floor(events_v[:, 3]).astype(np.int32),
                                  np.clip(np.floor(events_v[:, 2]).astype(np.int32), 0, self.time_num - 1),
                                  np.floor(events_v[:, 1]).astype(np.int32), np.floor(events_v[:, 0]).astype(np.int32)), 1)
            else:
                reprs = []
                for repr_name in self.repr:
                    repr_array = repr_map[repr_name](events_v[:, 2], events_v[:, 0].astype(np.int32),
                                                     events_v[:, 1].astype(np.int32),
                                                     events_v[:, 3], repr_size=(200, 200), time_num=self.time_num)
                    # standardization
                    # mu = np.mean(repr_array)
                    # sigma = np.std(repr_array)
                    # repr_array = (repr_array - mu) / sigma

                    # show
                    # anim = show_repr(repr_array)
                    # anim.save('anims/animation_' + str(idx) + '_label' + str(label) + '_' + repr_name + '.mp4', fps=100)

                    reprs.append(repr_array)
                reprs = np.array(reprs)
            reprs = torch.tensor(reprs, dtype=torch.float)
            # print(torch.max(reprs))
            # time to last dim
            reprs = reprs.permute(0, 2, 3, 1)
            reprs_views.append(reprs)
            labels.append(label)
        return reprs_views, labels




if __name__ == '__main__':
    # for THU-EACT-50
    data_directory = "/mnt/nas02nc/datasets/THU-EACT-50"
    repr = ['eventAccuFrame'] #, 'eventAccuFrame', 'timeSurface', 'eventCount']
    dataset = THU_EACT_50(path=data_directory, mode="front", train=True, augmentation=False, repr=repr)

    # for THU-EACT-50-CHL
    # data_directory = "H:/Event_camera_action/THU-EACT-50-CHL"
    # repr = ['timeSurface']
    # dataset = THU_EACT_50_CHL(datafile=data_directory, eval=True, augmentation=False, repr=repr)

    samples = 10
    for i in range(samples):
        index_to_test = i  # index of the sample you want to test
        single_sample_reprs, single_sample_label = dataset.__getitem__(index_to_test)

    # Output the results
    print("Representation Shape:", single_sample_reprs.shape)
    print("Label:", single_sample_label)
