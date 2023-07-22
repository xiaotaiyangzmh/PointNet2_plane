# Transfer pointnet2 to binary classification for plane and non-plane.

## Requirements
* Python 3.7
* Pytorch 1.13.1
* os
* sys
* argparse
* logging
* importlib
* datetime
* shutil
* pathlib
* tqdm
* numpy

## Installation
```
git clone https://github.com/xiaotaiyangzmh/PointNet2_plane.git
cd PointNet2_plane/
```

## Training
If the training uses transfer learning
Run the following command to train a plane segmentation model with all synthetic data (dataset 1 & dataset 2)
```
python train_synthetic_data.py --model pointnet2_sem_seg --datapath1 "./data_synthetic/pcd_plane" --datapath2 "./data_synthetic/pcd_nonplane" --epoch 32 --log_dir pointnet2_synthetic_data

```
Run the following command to train a plane segmentation model with real scene
```
python train_real_data.py --model pointnet2_sem_seg --datapath "./data_scene" --epoch 32 --log_dir pointnet2_real_data

```

If the model is trained from scratch, add one argument ```--transfer```, for example
```
python train_synthetic_data.py --model pointnet2_sem_seg --datapath1 "./data_synthetic/pcd_plane" --datapath2 "./data_synthetic/pcd_nonplane" --epoch 32 --log_dir pointnet2_synthetic_data --transfer

```

## Testing
Run the following command to test the model with the labelled whole scene
```
python test_labelled.py  --log_dir pointnet2_real_data --visual
python test_labelled.py  --log_dir pointnet2_synthetic_data --visual
```

Run the following command to test the model with the unlabelled whole scene
```
python test_unlabelled.py  --log_dir pointnet2_real_data
python test_unlabelled.py  --log_dir pointnet2_synthetic_data
```

## Reference
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

