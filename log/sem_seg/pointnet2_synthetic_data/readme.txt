Experiment description: 1497 plane and 310 non plane clouds for training, 167 plane and 34 non plane clouds for testing, 
4 ori example data and 3 lidar data collected in the lab for whole scene evaluation

Accuracy.png: accuracy during trainig

Mean Loss.png: mean loss during training

pointnet2_sem_seg.txt: training details in each epoch

eval.txt: evaluation details

eval_labelled: evaluation with 4 ori example data, containing gt labels, pred labels, gt colored pcd and pred colored pcd, 
using oxf package to label the point cloud. Blue for plane, pink for non-plane when visualizing the point cloud.

eval_unlabelled: evaluation with 3 lidar data collected in the lab, containing pred labels and pred colored pcd.
Blue for plane, pink for non-plane when visualizing the point cloud.


