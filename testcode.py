import os
import sys
import argparse
import logging
import importlib
import datetime
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d

from datasets import PointData
from models import pointnet2_sem_seg
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non-plane', 'plane']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--train_ratio', default=0.7, type=float, help='ratio of train set')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()

def main(args):
    def log_string(args_str):
        logger.info(args_str)
        print(args_str)

    # '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # experiment_dir = Path('./log/')
    # experiment_dir.mkdir(exist_ok=True)
    # experiment_dir = experiment_dir.joinpath('sem_seg')
    # experiment_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    #     experiment_dir = experiment_dir.joinpath(timestr)
    # else:
    #     experiment_dir = experiment_dir.joinpath(args.log_dir)
    # experiment_dir.mkdir(exist_ok=True)
    # checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = experiment_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)

    '''HYPER PARAMETER'''
    rootpath = "./data"
    num_classes = 2
    num_points = args.npoint
    batch_size = args.batch_size
    lr_net = args.learning_rate
    train_ratio = args.train_ratio
    epochs = args.epoch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_string("using {} device.".format(device))

    # '''DATASET LOADING'''
    # log_string("start loading training data ...")
    # train_set = PointData(rootpath, num_classes, num_points, 1.0, train_ratio, "train")
    # train_loader = DataLoader(train_set,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           pin_memory=True, 
    #                           drop_last=True,
    #                           num_workers=0)
    # log_string("using {} samples for training.".format(train_set.__len__()))
    # weights = torch.Tensor(train_set.labelweights).to(device)

    # log_string("start loading testing data ...")
    # test_set = PointData(rootpath, num_classes, num_points, 1.0, train_ratio, "test")
    # test_loader = DataLoader(test_set,
    #                           batch_size=batch_size,
    #                           shuffle=False,
    #                           pin_memory=True, 
    #                           drop_last=True,
    #                           num_workers=0)
    # log_string("using {} samples for testing.".format(test_set.__len__()))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_classes).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)
    
    # current model state dict
    classifier_state_dict = classifier.state_dict()
    # print(type(classifier.state_dict()), classifier.state_dict().keys())
    # print(classifier.state_dict()['sa1.mlp_convs.0.weight'].shape)

    # load pre-trained model
    transfer_model = torch.load('./weights/semseg_model.pth')
    # print(type(transfer_model['model_state_dict']), transfer_model['model_state_dict'].keys())

    for key in transfer_model['model_state_dict'].keys():
        if key in classifier_state_dict.keys():
            if transfer_model['model_state_dict'][key].shape == classifier_state_dict[key].shape:
                classifier_state_dict[key] = transfer_model['model_state_dict'][key]
    
    require_grad_layer = ["sa1", "conv2"]
    for param in classifier.named_parameters():
        if param[0][:3] in require_grad_layer or param[0][:5] in require_grad_layer:
            param[1].requires_grad = True
            print(param[0], param[1].shape)
        else:
            param[1].requires_grad = False
    grad_params = [p for p in classifier.parameters() if p.requires_grad]
    print(len(grad_params))

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

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/data_plane/pcd_nonplane/pcd_no_noise/2.pcd")
    # o3d.visualization.draw_geometries([pcd])

    # pcd = o3d.io.read_point_cloud("/home/minghan/workspace/plane_detection_NN/PointNet2_plane/data_plane/pcd_nonplane/pcd_noise/2.pcd")
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

    a = list([2, 3, 3]) + list([4, 5])
    idx = [2, 3]