import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

try:
    import joint_transformer as joint_transforms
except:
    import utils.joint_transformer as joint_transforms

class UWCOSDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        
        self.images = [os.path.join(image_root,f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        # self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip RandomCrop')
            self.joint_transform = joint_transforms.Compose([
                joint_transforms.RandomHorizontallyFlip(),
                joint_transforms.RandomVerticallyFlip(),
                joint_transforms.randomCrop(),
                joint_transforms.randomRotation(),
                joint_transforms.Resize((self.trainsize, self.trainsize)),
            ])
            self.img_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.target_transform = transforms.ToTensor()
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.edge_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])
        
        if self.augmentations:
            image, gt, edge = self.joint_transform(image, gt, edge)
            image = self.img_transform(image)
            gt = self.target_transform(gt)
            edge = self.target_transform(edge)
        else:
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
            edge = self.edge_transform(edge)
        return image, gt, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts) 
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and gt.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = UWCOSDataset(image_root, gt_root, edge_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root,f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root,f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
