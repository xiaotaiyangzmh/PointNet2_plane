import os
import re
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PointData(Dataset):
    """
    Arguments:

    rootpath: "./data"
    train_ratio: the number of files for training
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, train_ratio, mod, block_size=1.0):

        super(PointData, self).__init__()

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


class SceneData():
    def __init__(self, rootpath, num_classes, num_point, train_ratio, stride=0.5, block_size=1.0, padding=0.001, mod="test"):
        super(SceneData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.rootpath = rootpath
        self.mod = mod
        self.stride = stride

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
            self.split_indices = indices[:train_size]
        elif mod == "test":
            self.split_indices = indices[train_size:]
        else:
            raise Exception("mod should be train or test")
        print(f"loading {len(self.split_indices)} models ...")

        self.scene_points_num = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(self.split_indices, total=len(self.split_indices)):
            # load point cloud
            with open(cloud_files[i]) as f:
                lines = f.readlines()
            points_num = len(lines)
            points = np.zeros((points_num, 3))
            for j in range(points_num):
                point_str = re.split(",|\n", lines[j])
                point = np.asarray([float(n) for n in point_str[:-1]])
                points[j] = point

            # load labels
            with open(label_files[i]) as f:
                lines = f.readlines()
            # 0 for non-plane, 1 for plane
            l_labels = [int(l[0]) if int(l[0]) == 0 else 1 for l in lines]
            labels = np.asarray(l_labels)
            tmp, _ = np.histogram(labels, range(num_classes+1))
            labelweights += tmp

            self.scene_points_num.append(points.shape[0])
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(labels)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_scene, label_scene, sample_weight, index_scene = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.num_point))
                point_size = int(num_batch * self.num_point)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_scene = np.vstack([data_scene, data_batch]) if data_scene.size else data_batch
                label_scene = np.hstack([label_scene, label_batch]) if label_scene.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_scene.size else batch_weight
                index_scene = np.hstack([index_scene, point_idxs]) if index_scene.size else point_idxs
        data_scene = data_scene.reshape((-1, self.num_point, data_scene.shape[1]))
        label_scene = label_scene.reshape((-1, self.num_point))
        sample_weight = sample_weight.reshape((-1, self.num_point))
        index_scene = index_scene.reshape((-1, self.num_point))
        return data_scene, label_scene, sample_weight, index_scene

    def __len__(self):
        return len(self.scene_points_list)

