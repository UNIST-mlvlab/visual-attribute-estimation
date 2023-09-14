import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

# from tools.function import get_pkl_rootpath


class KetiAttr(data.Dataset):

    def __init__(self, pickle_root, image_root, split, transform=None, target_transform=None, idx=None):

        print("which pickle", pickle_root)

        dataset_info = pickle.load(open(pickle_root, 'rb+'))

        img_id = dataset_info["image_names"]

        attr_label = dataset_info["labels"]


        self.transform = transform
        self.target_transform = target_transform

        self.root_path = image_root


        self.attr_num = len(attr_label[0])
        print(f'{split} target_label: all')

        self.img_idx = dataset_info['partitions'][split]
        if idx is not None:
            self.img_idx = idx

        self.img_num = self.img_idx
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx].astype(float)  # [:, [0, 12]]
        #import ipdb;ipdb.set_trace()
    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]

        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname,  # noisy_weight

    def __len__(self):
        return len(self.img_id)

