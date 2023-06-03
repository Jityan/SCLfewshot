import os.path as osp
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

ROOT_PATH = "c:\\Users\\User\\Desktop\\scl\\data\\fc100\\"

class FC100(Dataset):

    def __init__(self, setname, size=32, transform=None):
        # Set the path according to train, val and test
        if setname=='train':
            THE_PATH = osp.join(ROOT_PATH, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = osp.join(ROOT_PATH, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = osp.join(ROOT_PATH, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if transform is None:
            if setname in ['train', 'trainval']:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                    ),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                        )
                ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class SSLFC100(Dataset):

    def __init__(self, setname, args):
        if setname=='train':
            THE_PATH = osp.join(ROOT_PATH, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = osp.join(ROOT_PATH, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = osp.join(ROOT_PATH, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.args = args

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(args.size, args.size)[-2:],
                        scale=(0.5, 1.0)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                )
        ])
        #
        self.identity_transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                )
        ])
        self.shared_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img = Image.open(path).convert('RGB')
        img = self.shared_transform(img)
        image = []
        for _ in range(1):
            image.append(self.identity_transform(img).unsqueeze(0))
        for i in range(3):
            image.append(self.augmentation_transform(img).unsqueeze(0))
        return dict(data=torch.cat(image)), label
