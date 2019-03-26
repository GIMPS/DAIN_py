from __future__ import absolute_import
import os
import os.path as osp
from utils.osutils import mkdir_if_missing
from PIL import Image
import cv2
import numpy as np



def search_in_512(fpath, fname):
    # working_dir = osp.dirname(osp.abspath(__file__))
    # high_res_path = osp.join(working_dir, 'data/GTOS_shape_reconstruction/material')
    # _fpath = fpath.split('/')[-1][2:]
    # high_res_path = osp.join(high_res_path, _fpath)
    # if not osp.isdir(high_res_path):
    #     return osp.join(fpath, fname), False
    # for file in os.listdir(high_res_path):
    #     if fname.split('_')[-1] in ['high.jpg','low.jpg', 'normal.jpg']:
    #         key_idx = int(fname.split('_')[-2][-2:])
    #         if file.split('_')[-2][0] != 'i':
    #             val_idx = int(file.split('_')[-2][:])
    #         else:
    #             val_idx = int(file.split('_')[-2][1:])
    #         if key_idx == val_idx:
    #             return osp.join(high_res_path, file), True
    #     else:
    #         if fname == file:
    #             return osp.join(high_res_path,file), True

    return osp.join(fpath,fname),False

def make_diff(fpath, diff_path):

    image_list = os.listdir(fpath)
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    neighbour_list = os.listdir(fpath)
    if '.DS_Store' in neighbour_list:
        neighbour_list.remove('.DS_Store')
    image_name = sorted(image_list)[1]
    neighbour_name= sorted(neighbour_list)[2]
    # image_name = sorted(os.listdir(fpath))[2]
    # neighbour_name= sorted(os.listdir(fpath))[3]
    image_path, image_found = search_in_512(fpath, image_name)
    neighbour_path, neighbour_found = search_in_512(fpath, neighbour_name)
    print('--------')
    print(image_path)
    # Affine Transform
    img_ = cv2.imread(image_path)
    # img_ = cv2.resize(img_, (512, 512))
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(neighbour_path)
    # img = cv2.resize(img, (512, 512))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.average(img1)
    mmax = np.amax(img1)
    mmin = np.amin(img1)
    variance = mmax - mmin
    # lb = mean - variance / 2
    # ub = mean + variance / 2

    is_align = False
    try:
        # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03,edgeThreshold=10000)
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        img3 = cv2.drawKeypoints(img1, kp1, img1)
        cv2.imwrite(diff_path+'/sift_keypoints1.png', img3)
        cv2.imwrite(diff_path+'/original1.png', img1)
        img3 = cv2.drawKeypoints(img2, kp2, img2)
        cv2.imwrite(diff_path+'/sift_keypoints2.png', img3)
        cv2.imwrite(diff_path+'/original2.png', img2)

        stereo = cv2.StereoBM_create()
        disparity = stereo.compute(img1, img2)
        cv2.imwrite(diff_path+'/disparity.png', disparity)

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

        dst = cv2.warpPerspective(img1, H, (img_.shape[1], img_.shape[0]))
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
        # diff =cv2.cvtColor(img-dst, cv2.COLOR_BGR2GRAY)
        diff = np.absolute(img2-dst)
        diff[idx] = mean
        # diff = cv2.normalize(diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # diff = cv2.bilateralFilter(diff, 10, 0, 0)
        # print(diff)
        # diff = cv2.equalizeHist(diff)
        # diff = cv2.medianBlur(diff,3)
        # diff = cv2.blur(diff, (3,3))
        # diff = cv2.GaussianBlur(diff, (3,3),0)
        # diff = cv2.bilateralFilter(diff, 9, 75, 75)
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
        diff = np.absolute(img2 - img1)

    # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.normalize(diff, None, mmin, mmax, norm_type=cv2.NORM_MINMAX)
    diff = cv2.resize(diff, (240, 240))
    diff = np.stack((diff,) * 3, axis=-1)
    cv2.imwrite(diff_path+'/diff.png',diff)

    return is_align



if __name__ == '__main__':
    working_dir = osp.dirname(osp.abspath(__file__))
    data_dir = "."
    dataset = 'tmp'
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
