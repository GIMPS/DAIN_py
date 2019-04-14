# DAIN_py

>    FYP Project -- Variance-aware Learning Based Material Recognition Using Stereo Cameras

## Overview

This repo includes 3 parts:
 1. Implementation of *Differential Angular Imaging for Material Recognition (2017 CVPR)* in  `main.py`
 2. Baseline model in `main_single_view.py`
 3. A model taking combinational input {original RGB, Differential Angular Image, Depth Map} in `main_depth.py`
3. Variance learning model in `main_var.py`

## To do list

 - [x] Reproduce with Pytorch 1.0
 - [x] Reproduce DAIN single view paper results
 - [x] Improve on DAIN baseline
 ~~- [ ] Reproduce 4D light field paper results~~ 
 - [x] Find a better fusion method
 - [x] Discuss whether differential imaging is neccessary
 - [x] Discuss ways to utilize depth features
 ~~- [ ] Export to a Caffe2 model with ONNX~~
 - [x] Collect a dataset with phone's dual cameras
 - [x] Make an Android app to demo this algorithm

## Block Issues

 - [x] Get access to dual cameras. Android P provides multi-camera APIs. However most phones do not support those API yet. ~~OnePlus~~ ~~POCO F1~~ **Huawei Mate20**

## General Issues

 - [x] Align DAIN dataset with calibration matrix. Find out if it is possible.

## Experiment Entries

 - Calibration Method: Sift Align

	**106**  images cannot be successfully aligned

 - Single View CNN

	|    Net    | Split |  Best Acc |
	|:---------:|:-----:|:---------:|
	| Resnet-50 |   1   |   81.0%   |
	| Resnet-50 |   2   |   84.0%   |
	| Resnet-50 |   3   |   80.6%   |
	| Resnet-50 |   4   |   84.7%   |
	| Resnet-50 |   5   |   85.5%   |

	Acc: 82.7


 - Single View DAIN

	|    Net    | Split |  Best Acc |
	|:---------:|:-----:|:---------:|
	| Resnet-50 |   1   |   82.4%   |
	| Resnet-50 |   2   |   83.3%   |
	| Resnet-50 |   3   |   80.9%   |
	| Resnet-50 |   4   |   84.2%   |
	| Resnet-50 |   5   |   86.7%   |

	Acc: 83.4
