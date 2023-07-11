import numpy as np
import re
import time
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ## script 1
    # pcd_file = "./data/cloud/fast_cloud.pcd"
    # normal_file = "./data/normal/normals.txt"
    # label_file = "./data/label/labels.txt"
    # plane_file = "./data/plane/planes.txt"

    # t0 = time.time()
    # pcd = o3d.io.read_point_cloud(pcd_file)
    # points = np.asarray(pcd.points)

    # t1 = time.time()
    # with open(normal_file) as f:
    #     lines = f.readlines()
    # normals = np.zeros((len(lines), 3))
    # for i in range(len(lines)):
    #     normal_str = re.split(",|\n", lines[i])
    #     normal = np.asarray([float(n) for n in normal_str[:-1]])
    #     normals[i] = normal

    # t2 = time.time()
    # with open(label_file) as f:
    #     lines = f.readlines()
    # label = np.asarray([int(l[0]) for l in lines]).reshape(-1, 1)

    # t3 = time.time()
    # with open(plane_file) as f:
    #     lines = f.readlines()
    # planes = dict()
    # for line in lines:
    #     line = line.split()
    #     plane_params = [float(l) for l in line[1:]]
    #     planes[int(line[0])] = plane_params

    # t4 = time.time()
    # print(t4-t3, t3-t2, t2-t1, t1-t0)

    ## script 2
    batch_size = 8
    train_set = datasets.pointData("./data", 4096, 0.8, "train")
    train_loader = DataLoader(train_set,
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=0)
    for points_data, labels_data, index in train_loader:
        print(type(points_data), points_data.shape)
        print(type(labels_data), labels_data.shape)

    # ## script 3
    # data = np.load("./data/HKPS_labels.npy")
    # print(data)

    # ## script 4
    # pcd_file = "./data/ouster/test.pcd"
    # pcd = o3d.io.read_point_cloud(pcd_file)
    # points = np.asarray(pcd.points)
    # print(torch.tensor(points))