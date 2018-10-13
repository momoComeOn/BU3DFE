import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os.path as osp
import numpy as np
import torch
import glob
import os
from PIL import Image


class BU3DFE(data.Dataset):
    def __init__(self, root, prefix, split='train', transform=True, flipcrop=False,image=224):
        self.root = root
        self.prefix = prefix
        self.split = split
        self._transform = transform
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if flipcrop:
            crop = transforms.RandomCrop(
                image) if self.split == 'train' else transforms.CenterCrop(image)
            flip = transforms.RandomHorizontalFlip() if self.split == 'train' else lambda x: x
            self.im_transform = transforms.Compose([
                transforms.Resize((256)),
                crop,
                flip,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                )
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.Resize((image, image)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                )
            ])

        self.data = None
        self.label = None
        self.get_data()
        # self.s_print()

    def s_print(self):
        print(self.data)
        print(self.label)

    def get_data(self):
        self.data = [root_.split(' ')[0] for root_ in self.root]
        self.label = [int(root_.strip().split(' ')[1]) for root_ in self.root]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        label = self.label[index]
        dirs = ['BU3DFE-2', 'BU3DFE-D', 'BU3DFE-NormalX',
                'BU3DFE-NormalY', 'BU3DFE-NormalZ', 'BU3DFE-SI']
        # label = [ self.label[index] for i in dirs]
        image_paths = [os.path.join(os.path.join(
            self.prefix, i, image_path)) for i in dirs]

        img = [Image.open(path) for path in image_paths]
        img = [i.convert('RGB') if i.mode == 'L' else i for i in img]
        return self.transform(img, label)

    def transform(self, img, lbl):
        new_img = [self.im_transform(i) for i in img]
        # new_img = torch.cat(new_img,dim=0)
        return new_img, lbl


def get_database(root, rate=0.9):
    with open(root, 'r') as root_file:
        lines = root_file.readlines()

    train_selet = ['F0001', 'F0002', 'F0003', 'F0004', 'F0005', 'F0006', 'F0007', 'F0008', 'F0009', 'F0010', 'F0011', 'F0012', 'F0013', 'F0014', 'F0015', 'F0016', 'F0017', 'F0018', 'F0019', 'F0020', 'F0021', 'F0022', 'F0023', 'F0024', 'F0025', 'F0026', 'F0027', 'F0028', 'F0029', 'F0030', 'F0031', 'F0032', 'F0033', 'F0034', 'F0035', 'F0036', 'F0037', 'F0038', 'F0039', 'F0040', 'F0041', 'F0042', 'F0043', 'F0044', 'F0045', 'F0046', 'F0047', 'F0048', 'F0049',
                   'F0050', 'F0051', 'F0052', 'F0053', 'F0054', 'F0055', 'F0056', 'M0001', 'M0002', 'M0003', 'M0004', 'M0005', 'M0006', 'M0007', 'M0008', 'M0009', 'M0010', 'M0011', 'M0012', 'M0013', 'M0014', 'M0015', 'M0016', 'M0017', 'M0018', 'M0019', 'M0020', 'M0021', 'M0022', 'M0023', 'M0024', 'M0025', 'M0026', 'M0027', 'M0028', 'M0029', 'M0030', 'M0031', 'M0032', 'M0033', 'M0034', 'M0035', 'M0036', 'M0037', 'M0038', 'M0039', 'M0040', 'M0041', 'M0042', 'M0043', 'M0044']
    import random
    random.shuffle(train_selet)

    train = train_selet[:int(len(train_selet)*rate)]
    text = train_selet[int(len(train_selet)*rate):]
    text_path, train_path = [], []
    for line in lines:
        if line[:5] in text:
            text_path.append(line)
        else:
            train_path.append(line)
    return train_path, text_path


def loader_bu3dfe(path, prefix, batch_size, shuffle=True, flipcrop=False,image=224):
    train_path, text_path = get_database(path)

    train_dataiter = BU3DFE(
        train_path, prefix, split='train', flipcrop=flipcrop,image=image)
    text_dataiter = BU3DFE(text_path, prefix, split='text', flipcrop=flipcrop,image=image)

    return data.DataLoader(train_dataiter, batch_size=batch_size, shuffle=shuffle), data.DataLoader(text_dataiter, batch_size=batch_size, shuffle=shuffle)

def loader_tensor(data_tensor,target_tensor,batch_size,shuffle=True):
    data_iter = data.TensorDataset(data_tensor,target_tensor)
    return data.DataLoader(data_iter,batch_size=batch_size,shuffle=shuffle)


if __name__ == '__main__':
    train_iter, text_iter = loader_bu3dfe(
        '../data/BU3DFE-2D/BU3DFE.txt', '/home/muyouhang/zkk/BU3DFE/data/BU3DFE-2D', 4)
    for i, (data, label) in enumerate(text_iter):
        print(i)
        print('text')
    for i, (data, label) in enumerate(train_iter):
        print(i)
        print('train')
