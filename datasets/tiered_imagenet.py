import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = 'c:\\Users\\User\\Desktop\\scl\\data\\tieredImageNet\\' # tiered-imagenet-kwon

class TieredImageNet(Dataset):
    def __init__(self, split, size=84, transform=None):
        data = np.load(os.path.join(
                ROOT_PATH, '{}_images.npz'.format(split)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]

        with open(os.path.join(
                ROOT_PATH, '{}_labels.pkl'.format(split)), 'rb') as f:
            label = pickle.load(f)['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        if transform is None:
            if split in ['train', 'trainval']:
                self.transform = transforms.Compose([
                    transforms.Resize(size+12),
                    transforms.RandomCrop(size, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]

class SSLTieredImageNet(Dataset):

    def __init__(self, split, args):
        data = np.load(os.path.join(
                ROOT_PATH, '{}_images.npz'.format(split)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]

        with open(os.path.join(
                ROOT_PATH, '{}_labels.pkl'.format(split)), 'rb') as f:
            label = pickle.load(f)['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1
        self.args = args

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
        self.augmentation_transform = transforms.Compose([transforms.RandomResizedCrop(size=(args.size, args.size)[-2:],
                        scale=(0.5, 1.0)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        #
        self.identity_transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        self.shared_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        img = self.shared_transform(img)
        image = []
        for _ in range(1):
            image.append(self.identity_transform(img).unsqueeze(0))
        for i in range(3):
            image.append(self.augmentation_transform(img).unsqueeze(0))
        return dict(data=torch.cat(image)), label
