from __future__ import absolute_import
import os.path as osp
import os
import numpy as np

from PIL import Image
import cv2
import torchvision.transforms as T

import random
class Preprocessor(object):
    """
    Base Preprocessor
    """
    def __init__(self, dataset, dataset_name, root=None, transform_img=None, transform_diff=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_img = transform_img
        self.transform_diff = transform_diff
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)


    def _get_single_item(self, index):
        fname, pid = self.dataset[index]
        fpath = fname

        if self.dataset_name == "GTOS_256":
            if self.root is not None:
                diff_path = osp.join(self.root[:-6]+'diff', fname)
                fpath = osp.join(self.root, fname)
            image_path = osp.join(fpath, sorted(os.listdir(fpath))[2])
            diff_path = osp.join(diff_path, 'diff.png')
            img = Image.open(image_path).convert('RGB')
            if self.transform_img is not None:
                img = self.transform_img(img)

            diff = Image.open(diff_path).convert('RGB')
            if self.transform_diff is not None:
                diff = self.transform_diff(diff)

            return img, diff, pid

        if self.dataset_name == "CDMS_174":
            if self.root is not None:
                fpath = osp.join(self.root, fname)
            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')
            diff_path = osp.join(fpath, 'DifferentialAngleImage.jpg')
            img = Image.open(image_path).convert('RGB')
            seed = random.randint(0, 2 ** 32)  # make a seed with numpy generator
            random.seed(seed)
            if self.transform_img is not None:
                img = self.transform_img(img)

            random.seed(seed)
            diff = Image.open(diff_path).convert('RGB')
            if self.transform_diff is not None:
                diff = self.transform_diff(diff)

            return img, diff, pid

        if self.dataset_name == "CDMS_174_diff":
            if self.root is not None:
                diff_path = osp.join(self.root[:-6] + 'diff', fname)
                fpath = osp.join(self.root, fname)
            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')
            diff_path = osp.join(diff_path, 'diff.png')
            img = Image.open(image_path).convert('RGB')
            seed = random.randint(0, 2 ** 32)  # make a seed with numpy generator
            random.seed(seed)
            if self.transform_img is not None:
                img = self.transform_img(img)

            random.seed(seed)
            diff = Image.open(diff_path).convert('RGB')
            if self.transform_diff is not None:
                diff = self.transform_diff(diff)

            return img, diff, pid
    

