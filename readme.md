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
Run the following command to start training
```
python train.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg
```


