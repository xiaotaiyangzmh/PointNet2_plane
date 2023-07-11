import os
import re
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class pointData(Dataset):
    """
    Arguments:

    rootpath: "./data"
    train_ratio: the number of files for training
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, block_size, train_ratio, mod):

        super(pointData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size

        # loading data
        cloud_path = os.path.join(rootpath, "cloud")
        label_path = os.path.join(rootpath, "label")
        cloud_filename = sorted(os.listdir(cloud_path))
        label_filename = sorted(os.listdir(label_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        # split data
        model_num = len(cloud_files)
        train_size = int(model_num * train_ratio)
        indices = list(range(model_num))
        random.seed(4)
        random.shuffle(indices)
        if mod == "train":
            split_indices = indices[:train_size]
        elif mod == "test":
            split_indices = indices[train_size:]
        else:
            raise Exception("mod should be train or test")
        print(f"loading {len(split_indices)} models ...")

        self.points_list, self.labels_list = [], []
        self.coord_min_list, self.coord_max_list = [], []
        num_point_all = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(split_indices, total=len(split_indices)):
            # load point cloud
            with open(cloud_files[i]) as f:
                lines = f.readlines()
            points_num = len(lines)
            points = np.zeros((points_num, 3))
            for j in range(points_num):
                point_str = re.split(",|\n", lines[j])
                point = np.asarray([float(n) for n in point_str[:-1]])
                points[j] = point
            # # normalize the point cloud
            # # move the minimum x, y, z to origin, and scale the point cloud inside a (1, 1, 1) box
            # points_min = np.amin(points, axis=0)
            # points = points - points_min
            # points_max = np.amax(points)
            # points = points / points_max

            # load labels
            with open(label_files[i]) as f:
                lines = f.readlines()
            # 0 for non-plane, 1 for plane
            l_labels = [int(l[0]) if int(l[0]) == 0 else 1 for l in lines]
            labels = np.asarray(l_labels)
            tmp, _ = np.histogram(labels, range(3))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.points_list.append(points), self.labels_list.append(labels)
            self.coord_min_list.append(coord_min), self.coord_max_list.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) / num_point)

        cloud_idxs = []
        for index in range(len(split_indices)):
            cloud_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.cloud_idxs = np.array(cloud_idxs)

        assert len(self.points_list) == len(self.labels_list)
        print(f"loading {len(self.points_list)} models successfully!")
        print(f"Totally {len(self.cloud_idxs)} samples in {mod} set.")
            
    def __getitem__(self, idx):
        cloud_idx = self.cloud_idxs[idx]
        points = self.points_list[cloud_idx]
        labels = self.labels_list[cloud_idx]
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            # increase the block scale to select appropriate number of points
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 3
        current_points = np.zeros((self.num_point, 6))
        # TODO: If we only want to distinguish plane and non-plane, we can scale the x, y, z differently,
        # but if we want to get the plane parameters, x, y, z should be scaled with the same value
        # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points)[:3]
        current_points[:, 3] = selected_points[:, 0] / self.coord_max_list[cloud_idx][0]
        current_points[:, 4] = selected_points[:, 1] / self.coord_max_list[cloud_idx][1]
        current_points[:, 5] = selected_points[:, 2] / self.coord_max_list[cloud_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):

        return len(self.cloud_idxs)

