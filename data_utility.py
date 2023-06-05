# -*- coding:utf-8 -*-
# __author__ = 'Vecchio'
import os
import glob
import numpy as np
import sys
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

rpm_folders = {'cs': "center_single",
               'io': "in_center_single_out_center_single",
               'ud': "up_center_single_down_center_single",
               'lr': "left_center_single_right_center_single",
               'd4': "distribute_four",
               'd9': "distribute_nine",
               '4c': "in_distribute_four_out_center_single",
               '*': '*'}


class dataset(Dataset):
    def __init__(self, args, mode, rpm_types):
        self.root_dir = args.path
        self.img_size = args.img_size
        self.set = args.dataset
        self.model = args.model
        self.mode = mode
        self.percent = args.percent if mode != "test" else 100
        self.shuffle_first = args.shuffle_first
        self.resize = transforms.Resize(self.img_size)
        self.transform = transforms.Compose([transforms.Resize(self.img_size),
                                             transforms.RandomHorizontalFlip(p=0.3),
                                             transforms.RandomVerticalFlip(p=0.3)])


        if self.set == "pgm":
            file_names = [f for f in os.listdir(self.root_dir) if mode in f]
            random.shuffle(file_names)
            self.file_names = file_names[:int(len(file_names) * self.percent / 100)]

        else:
            file_names = [[f for f in glob.glob(os.path.join(self.root_dir, rpm_folders[t], "*.npz")) if mode in f] for t in rpm_types]
            [random.shuffle(sublist) for sublist in file_names]
            file_names = [item for sublist in file_names for item in sublist[:int(len(sublist) * self.percent / 100)]]
            self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def shuffle(self, obj, pos):
        frames_o = []
        frames_p = []
        for f in zip(obj, pos):
            idx = torch.randperm(obj.size(1))
            frames_o.append(f[0][idx])
            frames_p.append(f[1][idx])
        obj = torch.stack(frames_o)
        pos = torch.stack(frames_p)
        return obj, pos

    def __getitem__(self, idx):
        if self.set == "pgm":
            data_path = self.root_dir + '/' + self.file_names[idx]
            data = np.load(data_path)
            images = data["image"].reshape(16, 160, 160)
            target = data["target"]

            # Shuffle choices.
            if self.mode == "train":
                context = images[:8]
                choices = images[8:]
                indices = list(range(8))
                np.random.shuffle(indices)
                new_target = indices.index(target)
                new_choices = choices[indices]
                images = np.concatenate((context, new_choices))
                images = self.transform(torch.tensor(images, dtype=torch.float32))
                target = new_target
            else:
                images = self.resize(torch.tensor(images, dtype=torch.float32))


            # Return tensors.
            return images, torch.tensor(target, dtype=torch.long)

        else:
            data_path = self.file_names[idx]
            data = np.load(data_path)
            target = data["target"]
            images = data["image"]

            # Shuffle choices to a) avoid exploiting any statistical bias in the dataset, and b) mitigate overfitting.
            if self.mode == "train":
                context = images[:8]
                choices = images[8:]
                idx = list(range(8))
                np.random.shuffle(idx)
                images = np.concatenate((context, choices[idx]))
                target = idx.index(target)
            images = torch.tensor(images, dtype=torch.float32)
            if self.mode == 'train':
                images = self.transform(images)
            else:
                images = self.resize(images)

            target = torch.tensor(target, dtype=torch.long)

            return images, target
