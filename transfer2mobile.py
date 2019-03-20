
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend




"""

python main_single_view.py --logs-dir Resnet50-single_view

"""

import argparse
import os.path as osp
import numpy as np

from utils.data import Dataset
from utils.data.preprocessor_single_view import Preprocessor
import torchvision.transforms as T
from torch.utils.data import DataLoader

import models
from torch import nn
import torch

from utils.serialization import load_checkpoint, save_checkpoint
from trainers_single_view import Trainer
from evaluators_single_view import Evaluator
import sys
from utils.logging import Logger

def get_data(name, split_id, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    dataset = Dataset(root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train
    num_classes = dataset.num_class

    train_transformer_img = T.Compose([
        # T.Resize((height, width)),
        T.ToTensor(),
        # normalizer,
    ])

    test_transformer_img = T.Compose([
        # T.Resize((height, width)),
        # T.RectScale(height, width),
        T.ToTensor(),
        # normalizer,
    ])



    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform_img=train_transformer_img),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)


    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform_img=test_transformer_img),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)



    return dataset, num_classes, train_loader, val_loader



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (240, 240)
    dataset, num_classes, train_loader, val_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)


    # Create model

    img_branch = models.create(args.arch, cut_layer=args.cut_layer, num_classes = num_classes)

    args.resume = "/mnt/lustre/renjiawei/DAIN_py/logs/Resnet50-single_view-split1/model_best.pth.tar"

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # img_high_level.load_state_dict(checkpoint['state_dict_img'])
        # diff_high_level.load_state_dict(checkpoint['state_dict_diff'])
        img_branch.load_state_dict(checkpoint['state_dict_img'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    img_branch = nn.DataParallel(img_branch).cuda()
    # img_branch = nn.DataParallel(img_branch)
    img_branch.train(False)

    x = torch.randn(64, 1, 224, 224, requires_grad=True)

    torch_out = torch.onnx._export( img_branch,  # model being run
                                   x,  # model input (or a tuple for multiple inputs)
                                   "super_resolution.onnx",
                                   # where to save the model (can be a file or file-like object)
                                   export_params=True)  # store the trained parameter weights inside the model file


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DAIN")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='GTOS_256')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cut-layer', type=str, default='layer2')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_save', type=int, default=2,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())








