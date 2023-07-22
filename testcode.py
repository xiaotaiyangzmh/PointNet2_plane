import os
import sys
import argparse
import logging
import importlib
import datetime
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d


if __name__ == "__main__":
    # args = parse_args()
    # main(args)


    # ## -------------- script 1 --------------
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

    # ## -------------- show pcd --------------
    # xyz = np.load("./data_plane/test/22.npy", allow_pickle=True)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz.item()['data'])
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/sem_seg/pointnet2_synthetic_data/eval_labelled/0_pred.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/sem_seg/pointnet2_synthetic_data/eval_labelled/1_pred.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/sem_seg/pointnet2_synthetic_data/eval_labelled/2_pred.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/log/sem_seg/pointnet2_synthetic_data/eval_labelled/3_pred.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/data_synthetic/pcd_combined_train/pcd_noise/2.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # label = np.load("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/data_plane/pcd_nonplane/label/2.npy")
    # print(label.shape, label[label==1].shape)

    # # loading data
    # rootpath = "./data_plane/pcd_data/pcd_noise"
    # cloud_filename = sorted(os.listdir(rootpath))
    # cloud_files = [os.path.join(rootpath, filename) for filename in cloud_filename]
    # i = 1313
    # pcd = o3d.io.read_point_cloud(cloud_files[i])
    # o3d.visualization.draw_geometries([pcd])

    # # # TODO: visualize with different colours
    # rootpath = "./data_plane/rawDataNonPlane"
    # # loading point cloud
    # cloud_path = os.path.join(rootpath, "cameraPC")
    # cloud_filename = sorted(os.listdir(cloud_path))
    # cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

    # # loading ground truth label
    # label_path = os.path.join(rootpath, "cameraLabels")
    # label_filename = sorted(os.listdir(label_path))
    # label_files = [os.path.join(label_path, filename) for filename in label_filename]
    # assert len(cloud_files) == len(label_files)

    # for i in range(len(cloud_files)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files[i])
    #     points_num = np.asarray(pcd.points).shape[0]

    #     # load labels
    #     labels = np.load(label_files[i])

    #     # paint the point cloud
    #     plane_colors = np.array([[0.1, 0.1, 0.3]])
    #     non_plane_colors = np.array([[0.8, 0.2, 0.3]])
    #     gt_colors = np.repeat(non_plane_colors, points_num, axis=0)
    #     gt_colors[labels==2] = plane_colors
    #     pcd.colors = o3d.Vector3dVector(gt_colors)
    #     o3d.visualization.draw_geometries([pcd])
    #     break

    # # TODO: visualize with different colours
    # root_path = "./log/sem_seg/pointnet2_with_nonplanes/visual"
    # all_filenames = sorted(os.listdir(root_path))
    # pred_files = [os.path.join(root_path, filename) for filename in all_filenames if filename.endswith("pred.pcd")]
    # gt_files = [os.path.join(root_path, filename) for filename in all_filenames if filename.endswith("gt.pcd")]
    # assert len(pred_files ) == len(gt_files)

    # for i in range(2):
    #     # load point cloud
    #     gt_pcd = o3d.io.read_point_cloud(gt_files[i])
    #     o3d.visualization.draw_geometries([gt_pcd])

    #     pred_pcd = o3d.io.read_point_cloud(pred_files[i])
    #     o3d.visualization.draw_geometries([pred_pcd])

    # # TODO: compute the percentage of planes and non-planes (gt)
    # rootpath = "./data_scene"
    # label_path = os.path.join(rootpath, "label")
    # label_filename = sorted(os.listdir(label_path))
    # label_files = [os.path.join(label_path, filename) for filename in label_filename]

    # gt_plane_num = 0
    # gt_nonplane_num = 0
    # for i in range(len(label_files)):
    #     # load labels
    #     with open(label_files[i]) as f:
    #         lines = f.readlines()
    #     # 0 for non-plane, 1 for plane
    #     l_labels = [int(l[0]) if int(l[0]) == 0 else 1 for l in lines]
    #     labels = np.asarray(l_labels)

    #     gt_plane_num += labels[labels == 1].shape[0]
    #     gt_nonplane_num += labels[labels == 0].shape[0]
    # print(f"plane is {round(gt_plane_num * 100 / (gt_plane_num + gt_nonplane_num), 2)}% of gt label")
    # print(f"non-plane is {round(gt_nonplane_num * 100 / (gt_plane_num + gt_nonplane_num), 2)}% of gt label")

    # # TODO: compute the percentage of planes and non-planes (pred with dataset 2)
    # pred_label_path = "./log/sem_seg/pointnet2_with_nonplanes/visual"
    # pred_label_filename = sorted(os.listdir(pred_label_path))
    # pred_label_files = [os.path.join(pred_label_path, filename) for filename in pred_label_filename if filename.endswith("pred.npy")]

    # pred_plane_num = 0
    # pred_nonplane_num = 0
    # for i in range(len(pred_label_files)):
    #     # load labels
    #     pred_labels = np.load(pred_label_files[i])
    #     pred_plane_num += pred_labels[pred_labels == 1].shape[0]
    #     pred_nonplane_num += pred_labels[pred_labels == 0].shape[0]
    # print(f"plane is {round(pred_plane_num * 100 / (pred_plane_num + pred_nonplane_num), 2)}% of pred label")
    # print(f"non-plane is {round(pred_nonplane_num * 100 / (pred_plane_num + pred_nonplane_num), 2)}% of pred label")


    # # TODO: build new scene
    # rootpath1 = "./data_synthetic/pcd_plane"
    # rootpath2 = "./data_synthetic/pcd_nonplane"
    # train_ratio1 = 0.9
    # train_ratio2 = 0.9
    # mod = "train"
    # # loading data with planes (dataset 1)
    # cloud_path1 = os.path.join(rootpath1, "pcd_noise")
    # cloud_filename1 = sorted(os.listdir(cloud_path1))
    # cloud_files1 = [os.path.join(cloud_path1, filename) for filename in cloud_filename1]
    
    # label_path1 = os.path.join(rootpath1, "label")
    # label_filename1 = sorted(os.listdir(label_path1))
    # label_files1 = [os.path.join(label_path1, filename) for filename in label_filename1]
    # assert len(cloud_files1) == len(label_files1)

    # # split data with planes (dataset 1)
    # model_num1 = len(cloud_files1)
    # train_size1 = int(model_num1 * train_ratio1)
    # indices1 = list(range(model_num1))
    # random.seed(4)
    # random.shuffle(indices1)
    # if mod == "train":
    #     split_indices1 = indices1[:train_size1][:10]
    # elif mod == "test":
    #     split_indices1 = indices1[train_size1:][:10]
    # else:
    #     raise Exception("mod should be train or test")
    # print(f"loading {len(split_indices1)} models ...")

    # plane_points_list = []
    # plane_label_list = []
    # for i in tqdm(split_indices1, total=len(split_indices1)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files1[i])
    #     points = np.asarray(pcd.points)
    #     plane_points_list.append(points)

    #     # load labels
    #     labels = np.load(label_files1[i]).astype(np.float64)
    #     plane_label_list.append(labels)

    # # loading data with non-planes (dataset 2)
    # cloud_path2 = os.path.join(rootpath2, "pcd_noise")
    # cloud_filename2 = sorted(os.listdir(cloud_path2))
    # cloud_files2 = [os.path.join(cloud_path2, filename) for filename in cloud_filename2]
    
    # label_path2 = os.path.join(rootpath2, "label")
    # label_filename2 = sorted(os.listdir(label_path2))
    # label_files2 = [os.path.join(label_path2, filename) for filename in label_filename2]
    # assert len(cloud_files2) == len(label_files2)

    # # split data with non-planes (dataset 2)
    # model_num2 = len(cloud_files2)
    # train_size2 = int(model_num2 * train_ratio2)
    # indices2 = list(range(model_num2))
    # random.seed(4)
    # random.shuffle(indices2)
    # if mod == "train":
    #     split_indices2 = indices2[:train_size2]
    # elif mod == "test":
    #     split_indices2 = indices2[train_size2:]
    # else:
    #     raise Exception("mod should be train or test")
    # print(f"loading {len(split_indices2)} models ...")

    # for i in tqdm(split_indices2, total=len(split_indices2)):
    #     # load point cloud
    #     pcd = o3d.io.read_point_cloud(cloud_files2[i])
    #     points = np.asarray(pcd.points)
    #     points = points - np.mean(points, axis=0)
    #     coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

    #     # load labels
    #     labels = np.load(label_files2[i]).astype(np.float64)

    #     # choose planes
    #     plane_idx = np.random.choice(np.arange(len(plane_points_list)), 5, replace=False)
    #     plane_points_0 = plane_points_list[plane_idx[0]]
    #     plane_points_1 = plane_points_list[plane_idx[1]]
    #     plane_points_2 = plane_points_list[plane_idx[2]]
    #     plane_points_3 = plane_points_list[plane_idx[3]]
    #     plane_points_4 = plane_points_list[plane_idx[4]]

    #     plane_labels_0 = plane_label_list[plane_idx[0]]
    #     plane_labels_1 = plane_label_list[plane_idx[1]]
    #     plane_labels_2 = plane_label_list[plane_idx[2]]
    #     plane_labels_3 = plane_label_list[plane_idx[3]]
    #     plane_labels_4 = plane_label_list[plane_idx[4]]

    #     offset_x = (coord_max[0]*0.5 - coord_min[0]) * np.random.random_sample() + coord_min[0]
    #     offset_y = (coord_max[1]*0.5 - coord_min[1]) * np.random.random_sample() + coord_min[1]
    #     plane_points_0 = plane_points_0 - np.mean(plane_points_0, axis=0) + np.array([0, 0, coord_max[2]])
    #     plane_points_1 = plane_points_1 - np.mean(plane_points_1, axis=0) + np.array([0, offset_y, coord_max[2]])
    #     plane_points_2 = plane_points_2 - np.mean(plane_points_2, axis=0) + np.array([offset_x, 0, coord_max[2]])
    #     plane_points_3 = plane_points_3 - np.mean(plane_points_3, axis=0) + coord_max
    #     plane_points_4 = plane_points_4 - np.mean(plane_points_4, axis=0) + coord_min

    #     combined_points = np.concatenate((points, plane_points_0, plane_points_1, plane_points_2, plane_points_3, plane_points_4))
    #     combined_labels = np.concatenate((labels, plane_labels_0, plane_labels_1, plane_labels_2, plane_labels_3, plane_labels_4))
    #     points_num = combined_points.shape[0]
    #     assert combined_points.shape[0] == combined_labels.shape[0]
    #     print((combined_points.shape[0] - points.shape[0]) / combined_points.shape[0])

    #     # colors
    #     plane_colors = np.array([[0.1, 0.1, 0.3]])
    #     non_plane_colors = np.array([[0.8, 0.2, 0.3]])
    #     combined_colors = np.repeat(non_plane_colors, points_num, axis=0)
    #     combined_colors[combined_labels==1] = plane_colors
    #     # create point cloud
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(combined_points)
    #     pcd.colors = o3d.Vector3dVector(combined_colors)
    #     o3d.visualization.draw_geometries([pcd])
    #     # break
