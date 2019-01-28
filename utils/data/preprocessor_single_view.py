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

    def __init__(self, dataset, root=None, transform_img=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_img = transform_img

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
        image_path = osp.join(fpath, sorted(os.listdir(fpath))[2])
        neighbour_path = osp.join(fpath, sorted(os.listdir(fpath))[3])

        # Affine Transform
        img_ = cv2.imread(image_path)

        img_ = Image.fromarray(img_.astype('uint8'), 'RGB')
        if self.transform_img is not None:
            img_ = self.transform_img(img_)

        return img_,pid



