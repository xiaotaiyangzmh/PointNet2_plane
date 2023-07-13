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
Run the following command to train a plane segmentation model using transfer learning
```
python train.py --model pointnet2_sem_seg --epoch 32 --log_dir pointnet2_sem_seg

```
Run the following command if the model is trained from scratch
```
python train.py --model pointnet2_sem_seg --epoch 32 --log_dir pointnet2_sem_seg_notrans --transfer

```

## Testing
Pre-trained model using transfer learning is saved in /log/sem_seg/pretained_model_1
Run the following command to test the pre-trained model with the whole scene
```
python test.py  --log_dir pretained_model_1 --visual
```
Or run the following command to test the model trained by yourself
```
python test.py  --log_dir pointnet2_sem_seg --visual
python test.py  --log_dir pointnet2_sem_seg_notrans --visual
```

## Reference
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

