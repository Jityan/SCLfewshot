# SCL: Self-supervised contrastive learning for few-shot image classification

This repository contains the **pytorch** code for the paper: "[SCL: Self-supervised contrastive learning for few-shot image classification](https://doi.org/10.1016/j.neunet.2023.05.037)" Jit Yan Lim, Kian Ming Lim, Chin Poo Lee, Yong Xuan Tan

\*Please note that we did not apply data augmentation to increase the support sample during the evaluation stage in this repository.

## Environment
The code is tested on Windows 10 with Anaconda3 and following packages:
- python 3.7.4
- pytorch 1.3.1

## Preparation
1. Change the ROOT_PATH value in the following files to yours:
    - `datasets/miniimagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`
    - `datasets/fc100.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - ***mini*ImageNet**: download from [CSS](https://github.com/anyuexuan/CSS) and put in `data/miniImageNet` folder.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tieredImageNet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifarfs` folder.

    - **FC100**: download from [MTL](https://github.com/yaoyao-liu/meta-transfer-learning), extract them into train, val, and test folders and put in `data/fc100` folder.

## Pre-trained Models
[Optional] The pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1S4cQZKiNl9zmANopI_3Og2be09HMaywk?usp=sharing). Extract and put the content in the `save` folder. To evaluate the model, run the `test.py` file with the proper save path as in the next section.

## Experiments
To train on miniImageNet:<br/>
```
python train.py --dataset mini --gamma-rot 1.5 --gamma-dist 0.02 --save-path ./save/mini-exp1
```
To evaluate on 5-way 1-shot and 5-way 5-shot miniImageNet:<br/>
```
python test.py --dataset mini --shot 1 --save-path ./save/mini-exp1
python test.py --dataset mini --shot 5 --save-path ./save/mini-exp1
```

## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@article{LIM2023,
  title = {SCL: Self-supervised contrastive learning for few-shot image classification},
  journal = {Neural Networks},
  year = {2023},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2023.05.037},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608023002812},
  author = {Jit Yan Lim and Kian Ming Lim and Chin Poo Lee and Yong Xuan Tan}
}
```

## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (jityan95@gmail.com) <br/>
Kian Ming Lim (Kian-Ming.Lim@nottingham.edu.cn)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[RFS](https://github.com/WangYueFt/rfs)**, and **[SKD](https://github.com/brjathu/SKD)**.
