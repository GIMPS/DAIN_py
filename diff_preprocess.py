from __future__ import absolute_import
import os
import os.path as osp
from utils.osutils import mkdir_if_missing
from PIL import Image
import cv2
import numpy as np




def make_diff(fpath, diff_path):
    # image_path = osp.join(fpath, sorted(os.listdir(fpath))[2])
    # neighbour_path= osp.join(fpath, sorted(os.listdir(fpath))[3])
    image_path = osp.join(fpath, 'NormalShot.jpg')
    neighbour_path= osp.join(fpath, 'WideShot.jpg')

    # Affine Transform
    img_ = cv2.imread(image_path)
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(neighbour_path)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_align = False
    try:
        # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03,edgeThreshold=10000)
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # img3 = cv2.drawKeypoints(img1, kp1, img1)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/sift_keypoints1.png', img3)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/original1.png', img1)
        # img3 = cv2.drawKeypoints(img2, kp2, img2)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/sift_keypoints2.png', img3)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/original2.png', img2)

        # stereo = cv2.StereoBM_create()
        # disparity = stereo.compute(img_, img)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/disparity.png', disparity)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m in matches:
            if m[0].distance < 0.5 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)

        if len(matches[:, 0]) < 4:
            raise AssertionError("Can’t find enough keypoints")

        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5)
        # save_path = "/Users/jason/Documents/GitHub/DAIN_py/tmp"
        # cv2.imwrite(save_path + '/view1.png', img_)
        # cv2.imwrite(save_path + '/view2.png', img)

        dst = cv2.warpPerspective(img_, H, (img_.shape[1], img_.shape[0]))
        # plt.imshow(img_)
        # plt.show()
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # plt.figure()
        # plt.imshow(dst)
        # plt.show()
        # plt.figure()
        idx = (dst == 0)
        img[idx] = 0
        # diff =cv2.cvtColor(img-dst, cv2.COLOR_BGR2GRAY)
        diff = img-dst
        # plt.imshow(diff)
        # plt.show()
        # plt.figure()

        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/crop1.png', dst)
        # cv2.imwrite('/Users/jason/Documents/GitHub/DAIN_py/tmp/crop2.png', img)
        # cv2.imwrite(save_path+'/diff.png',diff)
        # diff = cv2.bilateralFilter(diff, 9, 75, 75)
        is_align = True
    except:
        # diff = cv2.cvtColor(img - img_, cv2.COLOR_BGR2GRAY)
        diff = img - img_
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = np.stack((diff,) * 3, axis=-1)
    cv2.imwrite(diff_path+'/diff.png',diff)

    return is_align



if __name__ == '__main__':
    working_dir = osp.dirname(osp.abspath(__file__))
    data_dir = "/Users/jason/Desktop/FYP"
    dataset = 'CDMS_174_HQ'
    dataset_dir = osp.join(data_dir, dataset)
    img_dir = osp.join(dataset_dir, 'images')
    diff_dir = osp.join(dataset_dir, 'diff')
    mkdir_if_missing(diff_dir)
    cnt = 0
    for idx, scene in enumerate(sorted(os.listdir(img_dir))):
        img_scene_dir = osp.join(img_dir, scene)
        if not osp.isdir(img_scene_dir):
            continue
        diff_scene_dir = osp.join(diff_dir, scene)
        mkdir_if_missing(diff_scene_dir)
        if not make_diff(img_scene_dir, diff_scene_dir):
            cnt += 1

    print(cnt," images cannot be successfully aligned")
