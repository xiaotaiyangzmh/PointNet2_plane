import os
import re
import open3d as o3d
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SyntheticData(Dataset):
    """
    This class combine two dataset together.

    Arguments:
    
    rootpath1: root path of data with planes
    rootpath2: root path of data with non planes
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio1: the ratio of train and test, for data with planes
    train_ratio2: the ratio of train and test, for data with non planes
    mod: train or test
    """
    
    def __init__(self, rootpath1, rootpath2, num_classes, num_point, train_ratio1, train_ratio2, mod, block_size=1.0):

        super(SyntheticData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size

        self.points_list, self.labels_list = [], []
        self.coord_min_list, self.coord_max_list = [], []
        num_point_all = []
        labelweights = np.zeros(num_classes)

        # loading data with planes (dataset 1)
        cloud_path1 = os.path.join(rootpath1, "pcd_noise")
        cloud_filename1 = sorted(os.listdir(cloud_path1))
        cloud_files1 = [os.path.join(cloud_path1, filename) for filename in cloud_filename1]
        
        label_path1 = os.path.join(rootpath1, "label")
        label_filename1 = sorted(os.listdir(label_path1))
        label_files1 = [os.path.join(label_path1, filename) for filename in label_filename1]
        assert len(cloud_files1) == len(label_files1)

        # split data with planes (dataset 1)
        model_num1 = len(cloud_files1)
        train_size1 = int(model_num1 * train_ratio1)
        indices1 = list(range(model_num1))
        random.seed(4)
        random.shuffle(indices1)
        if mod == "train":
            split_indices1 = indices1[:train_size1]
        elif mod == "test":
            split_indices1 = indices1[train_size1:]
        else:
            raise Exception("mod should be train or test")
        print(f"loading {len(split_indices1)} models ...")

        for i in tqdm(split_indices1, total=len(split_indices1)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files1[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

            # load labels
            labels = np.load(label_files1[i]).astype(np.float64)
            tmp, _ = np.histogram(labels, range(3))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.points_list.append(points), self.labels_list.append(labels)
            self.coord_min_list.append(coord_min), self.coord_max_list.append(coord_max)
            num_point_all.append(labels.size)

        # loading data with non-planes (dataset 2)
        cloud_path2 = os.path.join(rootpath2, "pcd_noise")
        cloud_filename2 = sorted(os.listdir(cloud_path2))
        cloud_files2 = [os.path.join(cloud_path2, filename) for filename in cloud_filename2]
        
        label_path2 = os.path.join(rootpath2, "label")
        label_filename2 = sorted(os.listdir(label_path2))
        label_files2 = [os.path.join(label_path2, filename) for filename in label_filename2]
        assert len(cloud_files2) == len(label_files2)

        # split data with non-planes (dataset 2)
        model_num2 = len(cloud_files2)
        train_size2 = int(model_num2 * train_ratio2)
        indices2 = list(range(model_num2))
        random.seed(4)
        random.shuffle(indices2)
        if mod == "train":
            split_indices2 = indices2[:train_size2][:int(train_size2/3)]
        elif mod == "test":
            split_indices2 = indices2[train_size2:][:int((model_num2-train_size2)/3)]
        else:
            raise Exception("mod should be train or test")
        print(f"loading {len(split_indices2)} models ...")

        for i in tqdm(split_indices2, total=len(split_indices2)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files2[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

            # load labels
            labels = np.load(label_files2[i]).astype(np.float64)
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
        for index in range(len(split_indices1)):
            cloud_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        for index in range(len(split_indices2)):
            cloud_idxs.extend([index + len(split_indices1)] * int(round(sample_prob[index + len(split_indices1)] * num_iter)))
        self.cloud_idxs = np.array(cloud_idxs)

        assert len(self.points_list) == len(self.labels_list)
        print(f"loading {len(self.points_list)} models successfully!")
        print(f"Totally {len(self.cloud_idxs)} samples in {mod} set.")
            
    def __getitem__(self, idx):
        cloud_idx = self.cloud_idxs[idx]
        points = self.points_list[cloud_idx]
        labels = self.labels_list[cloud_idx]
        N_points = points.shape[0]

        # find the sampling center
        iter_num = 0
        tmp_size = self.block_size
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [tmp_size / 2.0, tmp_size / 2.0, 0]
            block_max = center + [tmp_size / 2.0, tmp_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
            else:
                iter_num += 1

            # increase the block scale if the center cannot be found
            if iter_num % 5 == 0:
                tmp_size += self.block_size

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize the sampled points
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
        selected_points[:, 2] = selected_points[:, 2] - center[2]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):

        return len(self.cloud_idxs)


class SyntheticCombinedData(Dataset):
    """
    This class create dataset with combined synthetic data.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    """
    
    def __init__(self, rootpath, num_classes, num_point, block_size=1.0):

        super(SyntheticCombinedData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size

        # loading data
        cloud_path = os.path.join(rootpath, "pcd_noise")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        label_path = os.path.join(rootpath, "label")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        # split data
        model_num = len(cloud_files)
        split_indices = list(range(model_num))
        print(f"loading {len(split_indices)} models ...")

        self.points_list, self.labels_list = [], []
        self.coord_min_list, self.coord_max_list = [], []
        num_point_all = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(split_indices, total=len(split_indices)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

            # load labels
            labels = np.load(label_files[i]).astype(np.float64)
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
        print(f"Totally {len(self.cloud_idxs)} samples.")
            
    def __getitem__(self, idx):
        cloud_idx = self.cloud_idxs[idx]
        points = self.points_list[cloud_idx]
        labels = self.labels_list[cloud_idx]
        N_points = points.shape[0]

        # find the sampling center
        iter_num = 0
        tmp_size = self.block_size
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [tmp_size / 2.0, tmp_size / 2.0, 0]
            block_max = center + [tmp_size / 2.0, tmp_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
            else:
                iter_num += 1

            # increase the block scale if the center cannot be found
            if iter_num % 5 == 0:
                tmp_size += self.block_size

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize the sampled points
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
        selected_points[:, 2] = selected_points[:, 2] - center[2]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):

        return len(self.cloud_idxs)


class RealData(Dataset):
    """
    This class create dataset with both planes and non-planes.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio: the ratio of train and test
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, train_ratio, mod, block_size=1.0):

        super(RealData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size

        # loading data
        cloud_path = os.path.join(rootpath, "cloud")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        label_path = os.path.join(rootpath, "label")
        label_filename = sorted(os.listdir(label_path))
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
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

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

        # find the sampling center
        iter_num = 0
        tmp_size = self.block_size
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [tmp_size / 2.0, tmp_size / 2.0, 0]
            block_max = center + [tmp_size / 2.0, tmp_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
            else:
                iter_num += 1

            # increase the block scale if the center cannot be found
            if iter_num % 5 == 0:
                tmp_size += self.block_size

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize the sampled points
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
        selected_points[:, 2] = selected_points[:, 2] - center[2]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):

        return len(self.cloud_idxs)


class SceneLabelledData():
    """
    This class create dataset with the whole scenes.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    train_ratio: the ratio of train and test
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, train_ratio=0.7, stride=0.5, block_size=1.0, padding=0.001, split=False):
        super(SceneLabelledData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.rootpath = rootpath
        self.stride = stride

        # loading data
        cloud_path = os.path.join(rootpath, "cloud")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        label_path = os.path.join(rootpath, "label")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        # split data
        model_num = len(cloud_files)
        indices = list(range(model_num))
        random.seed(4)
        random.shuffle(indices)
        if split == True:
            train_size = int(model_num * train_ratio)
            self.split_indices = indices[train_size:]
        else:
            self.split_indices = indices
        print(f"loading {len(self.split_indices)} models ...")

        self.scene_points_num = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        labelweights = np.zeros(num_classes)

        for i in tqdm(self.split_indices, total=len(self.split_indices)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

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


class SceneUnlabelledData():
    """
    This class create dataset without labeling.
    
    Arguments:

    rootpath: root path of data, "./data_scene"
    num_classes: the number of classes
    num_point: the number of points in each sampling group
    mod: train or test
    """
    
    def __init__(self, rootpath, num_classes, num_point, stride=1.5, block_size=3, padding=0.001, mod="test"):
        super(SceneUnlabelledData, self).__init__()

        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.rootpath = rootpath
        self.mod = mod
        self.stride = stride

        # loading data
        cloud_filename = sorted(os.listdir(rootpath))
        cloud_files = [os.path.join(rootpath, filename) for filename in cloud_filename]
        print(f"loading {len(cloud_files)} models ...")

        self.indices = range(len(cloud_files))
        self.scene_points_num = []
        self.scene_points_list = []

        for i in tqdm(self.indices, total=len(cloud_files)):
            # load point cloud
            pcd = o3d.io.read_point_cloud(cloud_files[i])
            points = np.asarray(pcd.points)
            points_num = points.shape[0]

            self.scene_points_num.append(points.shape[0])
            self.scene_points_list.append(points)
        assert len(self.scene_points_num) == len(self.scene_points_list)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_scene, index_scene = np.array([]), np.array([])
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

                data_scene = np.vstack([data_scene, data_batch]) if data_scene.size else data_batch
                index_scene = np.hstack([index_scene, point_idxs]) if index_scene.size else point_idxs
        data_scene = data_scene.reshape((-1, self.num_point, data_scene.shape[1]))
        index_scene = index_scene.reshape((-1, self.num_point))

        return data_scene, index_scene

    def __len__(self):
        return len(self.scene_points_list)