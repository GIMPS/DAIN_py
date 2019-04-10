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

    def __init__(self, dataset, dataset_name, root=None, transform_img=None, transform_diff=None, transform_depth = None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_img = transform_img
        self.transform_diff = transform_diff
        self.transform_depth = transform_depth
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

        # if self.dataset_name == "GTOS_256":
        #     if self.root is not None:
        #         diff_path = osp.join(self.root[:-6] + 'diff', fname)
        #         fpath = osp.join(self.root, fname)
        #     image_path = osp.join(fpath, sorted(os.listdir(fpath))[2])
        #     diff_path = osp.join(diff_path, 'diff.png')
        #     img = Image.open(image_path).convert('RGB')
        #     if self.transform_img is not None:
        #         img = self.transform_img(img)
        #
        #     diff = Image.open(diff_path).convert('RGB')
        #     if self.transform_diff is not None:
        #         diff = self.transform_diff(diff)
        #
        #     return img, diff, pid

        if self.dataset_name == "CDMS_174":
            # if self.root is not None:
            #     diff_path = osp.join(self.root[:-6] + 'diff', fname)
            #     fpath = osp.join(self.root, fname)
            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')
            diff_path = osp.join(fpath, 'DifferentialAngleImage.jpg')
            # diff_path = osp.join(diff_path, 'diff.png')
            depth_path = osp.join(fpath, 'DisparityMap2.jpg')

            seed = random.randint(0, 2 ** 32)  # make a seed with numpy generator

            img = Image.open(image_path).convert('RGB')
            random.seed(seed)
            if self.transform_img is not None:
                img = self.transform_img(img)

            random.seed(seed)
            diff = Image.open(diff_path).convert('RGB')
            if self.transform_diff is not None:
                diff = self.transform_diff(diff)

            random.seed(seed)
            depth= Image.open(depth_path).convert('RGB')
            if self.transform_depth is not None:
                depth = self.transform_depth(depth)

            return img, diff, depth, pid




        if self.dataset_name == "CDMS_160":
            if self.root is not None:
                diff_path = osp.join(self.root[:-6] + 'diff', fname)
                fpath = osp.join(self.root, fname)
            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')
            # diff_path = osp.join(fpath, 'DifferentialAngleImage.jpg')
            diff_path = osp.join(diff_path, 'diff.png')
            depth_path = osp.join(fpath, 'DisparityMapFilteredNormalized.jpg')

            seed = random.randint(0, 2 ** 32)  # make a seed with numpy generator

            img = Image.open(image_path).convert('RGB')
            random.seed(seed)
            if self.transform_img is not None:
                img = self.transform_img(img)

            random.seed(seed)
            diff = Image.open(diff_path).convert('RGB')
            if self.transform_diff is not None:
                diff = self.transform_diff(diff)

            random.seed(seed)
            depth= Image.open(depth_path).convert('RGB')
            if self.transform_depth is not None:
                depth = self.transform_depth(depth)

            return img, diff, depth, pid



