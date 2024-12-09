# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/datasets.py
"""

import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class BRATS20Dataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData' if is_train else 'MICCAI_BraTS2020_ValidationData')
        self.patients = [p for p in os.listdir(self.data_dir) if p.startswith('BraTS20_Training')]
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_path = os.path.join(self.data_dir, self.patients[idx])

        # Load modalities
        flair = nib.load(os.path.join(patient_path, f"{self.patients[idx]}_flair.nii")).get_fdata()
        t1 = nib.load(os.path.join(patient_path, f"{self.patients[idx]}_t1.nii")).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, f"{self.patients[idx]}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(patient_path, f"{self.patients[idx]}_t2.nii")).get_fdata()
        seg = nib.load(os.path.join(patient_path, f"{self.patients[idx]}_seg.nii")).get_fdata()

        # Normalize images
        flair = (flair - np.mean(flair)) / np.std(flair)
        t1 = (t1 - np.mean(t1)) / np.std(t1)
        t1ce = (t1ce - np.mean(t1ce)) / np.std(t1ce)
        t2 = (t2 - np.mean(t2)) / np.std(t2)

        # Stack modalities as channels
        image = np.stack([flair, t1, t1ce, t2], axis=0)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.long)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, seg
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)  # Add data augmentation if necessary
    dataset = BRATS20Dataset(data_dir=args.data_path, is_train=is_train, transform=transform)
    nb_classes = 2  # E.g., Tumor vs Non-tumor (binary segmentation)
    return dataset, nb_classes

def build_transform(is_train, args):
    # Mean and std for MRIs (update these with actual values if known)
    BRATS_MEAN = [0.5]
    BRATS_STD = [0.5]

    if is_train:
        # Apply data augmentation for 3D medical images
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),  # Randomly rotate
            transforms.RandomHorizontalFlip(),     # Flip horizontally
            transforms.RandomVerticalFlip(),       # Flip vertically
            transforms.ToTensor(),                 # Convert to tensor
            transforms.Normalize(BRATS_MEAN, BRATS_STD)  # Normalize
        ])
    else:
        # Validation transform (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(BRATS_MEAN, BRATS_STD)
        ])
    
    return transform

# class INatDataset(ImageFolder):
#     def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
#                  category='name', loader=default_loader):
#         self.transform = transform
#         self.loader = loader
#         self.target_transform = target_transform
#         self.year = year
#         # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
#         path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
#         with open(path_json) as json_file:
#             data = json.load(json_file)

#         with open(os.path.join(root, 'categories.json')) as json_file:
#             data_catg = json.load(json_file)

#         path_json_for_targeter = os.path.join(root, f"train{year}.json")

#         with open(path_json_for_targeter) as json_file:
#             data_for_targeter = json.load(json_file)

#         targeter = {}
#         indexer = 0
#         for elem in data_for_targeter['annotations']:
#             king = []
#             king.append(data_catg[int(elem['category_id'])][category])
#             if king[0] not in targeter.keys():
#                 targeter[king[0]] = indexer
#                 indexer += 1
#         self.nb_classes = len(targeter)

#         self.samples = []
#         for elem in data['images']:
#             cut = elem['file_name'].split('/')
#             target_current = int(cut[2])
#             path_current = os.path.join(root, cut[0], cut[2], cut[3])

#             categors = data_catg[target_current]
#             target_current_true = targeter[categors[category]]
#             self.samples.append((path_current, target_current_true))

#     # __getitem__ and __len__ inherited from ImageFolder
    
# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     if args.data_set == 'CIFAR10':
#         dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
#         nb_classes = 10
#     elif args.data_set == 'CIFAR100':
#         dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
#         nb_classes = 100
#     elif args.data_set == 'IMNET':
#         root = os.path.join(args.data_path, 'train' if is_train else 'val')
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif args.data_set == 'INAT':
#         dataset = INatDataset(args.data_path, train=is_train, year=2018,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes
#     elif args.data_set == 'INAT19':
#         dataset = INatDataset(args.data_path, train=is_train, year=2019,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes

#     return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         size = int(args.crop_ratio * args.input_size)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)
