# DAIN_py

>    FYP Project. Use differential angle information to do material
> classification. Apply to dual cameras on mobile phones.

## To do list

 - [x] Reproduce with Pytorch 1.0
 - [x] Reproduce DAIN single view paper results
 - [x] Improve on DAIN baseline
 - [ ] Reproduce 4D light field paper results 
 - [ ] Find a better fusion method
 - [ ] Discuss whether differential imaging is neccessary
 - [ ] Discuss ways to utilize depth features
 - [ ] Export to a Caffe2 model with ONNX
 - [ ] Collect a dataset with phone's dual cameras
 - [ ] Make an Android app to demo this algorithm

## Block Issues

 - [ ] Get access to dual cameras. Android P provides multi-camera APIs. However most phones do not support those API yet. ~~OnePlus~~ ~~POCO F1~~

## General Issues

 - [ ] Align DAIN dataset with calibration matrix. Find out if it is possible.

## Experiment Entries

 - Calibration Method: Sift Align

	**106**  images cannot be successfully aligned

 - Single View CNN

	|    Net    | Split |  Best Acc |
	|:---------:|:-----:|:---------:|
	| Resnet-50 |   1   |   85.9%   |
	| Resnet-50 |   2   |   89.1%   |
	| Resnet-50 |   3   |   82.8%   |
	| Resnet-50 |   4   |   93.8%   |
	| Resnet-50 |   5   |   89.1%   |

	`Acc: 88.3±5.5%  Mean Acc: 88.1%`


 - Single View DAIN

	|    Net    | Split |  Best Acc |
	|:---------:|:-----:|:---------:|
	| Resnet-50 |   1   |   81.2%   |
	| Resnet-50 |   2   |   90.6%   |
	| Resnet-50 |   3   |   87.5%   |
	| Resnet-50 |   4   |   95.3%   |
	| Resnet-50 |   5   |   92.2%   |

	`Acc: 89.36% Mean Acc: 88.3±7.1%`
