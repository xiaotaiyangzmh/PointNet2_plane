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
Run the following command to train a plane segmentation model with only plane data (dataset 1)
```
python train_only_planes.py --model pointnet2_sem_seg --datapath "./data_plane/pointcloud" --epoch 32 --log_dir pointnet2_only_planes

```
Run the following command to train a plane segmentation model with both planes and non-planes (dataset 2)
```
python train_with_nonplanes.py --model pointnet2_sem_seg --datapath "./data_scene" --epoch 32 --log_dir pointnet2_with_nonplanes

```
Run the following command to train a plane segmentation model with both planes and non-planes (dataset 1 & dataset 2)
```
python train_combine_data.py --model pointnet2_sem_seg --datapath1 "./data_plane/pointcloud" --datapath2 "./data_scene" --epoch 32 --log_dir pointnet2_combine_data

```

If the model is trained from scratch, add one argument ```--transfer```, for example
```
python train_with_nonplanes.py --model pointnet2_sem_seg --datapath "./data_scene" --epoch 32 --log_dir pointnet2_with_nonplanes --transfer

```

## Testing
Run the following command to test the model with the whole scene
```
python test.py  --log_dir pointnet2_only_planes --visual
python test.py  --log_dir pointnet2_with_nonplanes --visual
python test.py  --log_dir pointnet2_combine_data --visual
```

## Reference
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

