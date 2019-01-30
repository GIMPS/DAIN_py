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
    def __init__(self, dataset, root=None, transform_img=None, transform_diff=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_img = transform_img
        self.transform_diff = transform_diff

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
            diff_path = osp.join(self.root[:-6]+'diff', fname)
            fpath = osp.join(self.root, fname)
        image_path = osp.join(fpath, sorted(os.listdir(fpath))[2])
        diff_path = osp.join(diff_path, 'diff.png')
        # neighbour_path= osp.join(fpath, sorted(os.listdir(fpath))[3])

        # Affine Transform
        # img_ = cv2.imread(image_path)
        # img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # img = cv2.imread(neighbour_path)
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # try:
        #     # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03,edgeThreshold=10000)
        #     sift = cv2.xfeatures2d.SIFT_create()
        #     # find the keypoints and descriptors with SIFT
        #     kp1, des1 = sift.detectAndCompute(img1, None)
        #     kp2, des2 = sift.detectAndCompute(img2, None)
        #
        #     # img3 = cv2.drawKeypoints(img1, kp1, img1)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/sift_keypoints1.png', img3)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/original1.png', img1)
        #     # img3 = cv2.drawKeypoints(img2, kp2, img2)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/sift_keypoints2.png', img3)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/original2.png', img2)
        #
        #     # stereo = cv2.StereoBM_create()
        #     # disparity = stereo.compute(img_, img)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/disparity.png', disparity)
        #
        #     bf = cv2.BFMatcher()
        #     matches = bf.knnMatch(des1, des2, k=2)
        #
        #     matches=np.asarray(matches)
        #     src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        #     dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        #     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5)
        #     # save_path = "/Users/jason/Documents/GitHub/DAIN_py/tmp"
        #     # cv2.imwrite(save_path + '/view1.png', img_)
        #     # cv2.imwrite(save_path + '/view2.png', img)
        #
        #     dst = cv2.warpPerspective(img_, H, (img_.shape[1], img_.shape[0]))
        #     # plt.imshow(img_)
        #     # plt.show()
        #     # plt.figure()
        #     # plt.imshow(img)
        #     # plt.show()
        #     # plt.figure()
        #     # plt.imshow(dst)
        #     # plt.show()
        #     # plt.figure()
        #     idx = (dst == 0)
        #     img[idx] = 0
        #     # diff =cv2.cvtColor(img-dst, cv2.COLOR_BGR2GRAY)
        #     diff = img-dst
        #     # plt.imshow(diff)
        #     # plt.show()
        #     # plt.figure()
        #
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/crop1.png', dst)
        #     # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/crop2.png', img)
        #     # cv2.imwrite(save_path+'/diff.png',diff)
        #     # diff = cv2.bilateralFilter(diff, 9, 75, 75)
        # except:
        #     # diff = cv2.cvtColor(img - img_, cv2.COLOR_BGR2GRAY)
        #     # print(fname)
        #     diff = img - img_
        img = Image.open(image_path).convert('RGB')
        if self.transform_img is not None:
            img = self.transform_img(img)

        diff = Image.open(diff_path).convert('RGB')
        if self.transform_diff is not None:
            diff = self.transform_diff(diff)
        return img, diff, pid
    
    

