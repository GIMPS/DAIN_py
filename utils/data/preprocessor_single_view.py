from __future__ import absolute_import
import os.path as osp
import os
import numpy as np

from PIL import Image
import cv2
import torchvision.transforms as T


class Preprocessor(object):
    """
    Base Preprocessor
    """

    def __init__(self, dataset,  dataset_name, root=None, transform_img=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_img = transform_img
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
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        if self.dataset_name == "GTOS_256":

            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')

            img = Image.open(image_path).convert('RGB')

            if self.transform_img is not None:
                img = self.transform_img(img)

            return img,pid

        if self.dataset_name == "CDMS_174":

            image_path = osp.join(fpath, 'RectifiedNormalShot.jpg')

            img = Image.open(image_path).convert('RGB')

            if self.transform_img is not None:
                img = self.transform_img(img)

            return img,  pid



