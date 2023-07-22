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
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import SceneUnlabelledData
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


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/eval_unlabelled'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''HYPER PARAMETER'''
    rootpath = "./data_scene/real_data"
    num_classes = 2
    num_points = args.npoint
    batch_size = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_string("using {} device.".format(device))

    '''DATASET LOADING'''
    log_string("start loading testing data ...")
    test_scene_set = SceneUnlabelledData(rootpath, num_classes, num_points)
    log_string("using {} scene for testing.".format(test_scene_set.__len__()))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_classes).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = test_scene_set.indices
        num_batches = len(test_scene_set)

        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        total_iou_deno_class = [0 for _ in range(num_classes)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, str(scene_id[batch_idx])))

            whole_scene_data = test_scene_set.scene_points_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_data.shape[0], num_classes))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_point_index = test_scene_set[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + batch_size - 1) // batch_size
                batch_data = np.zeros((batch_size, num_points, 6))

                batch_label = np.zeros((batch_size, num_points))
                batch_point_index = np.zeros((batch_size, num_points))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * batch_size
                    end_idx = min((sbatch + 1) * batch_size, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            # save the label
            pred_filename = os.path.join(visual_dir, str(scene_id[batch_idx]) + '_pred.npy')
            np.save(pred_filename, pred_label)

            # save the point cloud
            plane_colors = np.array([[0.1, 0.1, 0.3]])
            non_plane_colors = np.array([[0.8, 0.2, 0.3]])
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(whole_scene_data)
            points_num = whole_scene_data.shape[0]
            # paint the pred point cloud
            pred_colors = np.repeat(non_plane_colors, points_num, axis=0)
            pred_colors[pred_label==1] = plane_colors
            pred_pcd.colors = o3d.Vector3dVector(pred_colors)
            # save the point cloud
            pcd_path = os.path.join(visual_dir, str(scene_id[batch_idx]) + '_pred.pcd')
            o3d.io.write_point_cloud(pcd_path, pred_pcd)

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)