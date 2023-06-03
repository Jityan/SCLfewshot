# SCL: Self-supervised contrastive learning for few-shot image classification

This repository contains the **pytorch** code for the paper: "[SCL: Self-supervised contrastive learning for few-shot image classification](https://doi.org/10.1016/j.neunet.2023.05.037)" Jit Yan Lim, Kian Ming Lim, Chin Poo Lee, Yong Xuan Tan

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
    - ***mini*ImageNet**: download from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE and put in `data/miniImageNet/images` folder. The csv split files can be downloaded from [here](https://github.com/yinboc/prototypical-network-pytorch) and put in `data/miniImageNet`.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tieredImageNet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifarfs` folder.

    - **FC100**: download from [MTL](https://github.com/yaoyao-liu/meta-transfer-learning), extract them into train, val, and test folders and put in `data/fc100` folder.

## Pre-trained Models
[Optional] The pre-trained models can be downloaded from [here](https://drive.google.com/file/d/1sH4dgqKhE9jfhediuL9Lf-zLFjHOk-2o/view?usp=sharing).

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
  author = {Jit Yan Lim and Kian Ming Lim and Chin Poo Lee and Yong Xuan Tan},
  keywords = {Few-shot learning, Self-supervised learning, Meta-learning, Contrastive learning},
  abstract = {Few-shot learning aims to train a model with a limited number of base class samples to classify the novel class samples. However, to attain generalization with a limited number of samples is not a trivial task. This paper proposed a novel few-shot learning approach named Self-supervised Contrastive Learning (SCL) that enriched the model representation with multiple self-supervision objectives. Given the base class samples, the model is trained with the base class loss. Subsequently, contrastive-based self-supervision is introduced to minimize the distance between each training sample with their augmented variants to improve the sample discrimination. To recognize the distant sample, rotation-based self-supervision is proposed to enable the model to learn to recognize the rotation degree of the samples for better sample diversity. The multitask environment is introduced where each training sample is assigned with two class labels: base class label and rotation class label. Complex augmentation is put forth to help the model learn a deeper understanding of the object. The image structure of the training samples are augmented independent of the base class information. The proposed SCL is trained to minimize the base class loss, contrastive distance loss, and rotation class loss simultaneously to learn the generic features and improve the novel class performance. With the multiple self-supervision objectives, the proposed SCL outperforms state-of-the-art few-shot approaches on few-shot image classification benchmark datasets.}
}
```

## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (lim.jityan@mmu.edu.my) <br/>
Kian Ming Lim (kmlim@mmu.edu.my)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[RFS](https://github.com/WangYueFt/rfs)**, and **[SKD](https://github.com/brjathu/SKD)**.
