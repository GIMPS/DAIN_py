import argparse
import os.path as osp
import numpy as np
import utils
from utils.data import Dataset
from utils.data.preprocessor import Image_Preprocessor, Diff_Preprocessor, Preprocessor
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import models
from torch import nn
import torch
from utils.meters import AverageMeter
import time

def show(img):
    npimg = img.numpy()
    plt.imshow((np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8))
    plt.show()

def show_flow(img):
    npimg = img.numpy()
    plt.imshow((np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8).squeeze(axis=2), cmap='gray')
    plt.show()

def get_data(name, split_id, data_dir, batch_size, workers):
    root = osp.join(data_dir, name)
    dataset = Dataset(root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train
    num_classes = 39

    train_transformer_img = T.Compose([
        T.Resize(224),
        # T.RandomSizedRectCrop(height, width),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        # normalizer,
    ])

    train_transformer_flow = T.Compose([
        T.Resize(224),
        # T.RandomSizedRectCrop(height, width),
        # T.RandomHorizontalFlip(),
        # T.Grayscale(),
        T.ToTensor(),
        # normalizer,
    ])

    test_transformer_img = T.Compose([
        T.Resize(224),
        # T.RectScale(height, width),
        T.ToTensor(),
        # normalizer,
    ])

    test_transformer_flow = T.Compose([
        T.Resize(224),
        # T.RectScale(height, width),
        T.Grayscale(),
        T.ToTensor(),
        # normalizer,
    ])
    #
    # a = Preprocessor(train_set, root=dataset.images_dir,
    #              transform_img=train_transformer_img, transform_flow=train_transformer_flow)
    # p = a[716]
    # show(p[0])
    # show_flow(p[1])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform_img=train_transformer_img, transform_flow=train_transformer_flow),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)


    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform_img=test_transformer_img,transform_flow=test_transformer_flow),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)



    return dataset, num_classes, train_loader, val_loader



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, feature_extractor, img_high_level, diff_high_level, criterion, img_optimizer, diff_optimizer , epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    img_high_level.train()
    diff_high_level.train()

    end = time.time()
    for i, (img, diff, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.float()
        diff = diff.float()
        # target = target.cuda(async=True)
        img_var = torch.autograd.Variable(img)
        diff_var = torch.autograd.Variable(diff)
        target_var = torch.autograd.Variable(target)

        img_feature = feature_extractor(img_var)
        diff_feature = feature_extractor(diff_var)

        print(img_feature.shape, "img_feature size")
        # print(diff_feature.shape, "diff_feature size")

        # sum
        fusion_feature = img_feature+diff_feature

        output = (img_high_level(img_feature)+diff_high_level(fusion_feature))/2

        # output =  img_high_level(img_feature)
        print(output.data, 'output val')
        print(target_var, 'target val')
        print(output.shape, "output size")
        print(target_var.shape, "target size")
        loss = criterion(output, target_var)
        print(loss.item(), "loss val")

        print(output.shape, "loss size")

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        print(prec1.data, prec3.data, "precision")
        losses.update(loss.item(), img.size(0))
        top1.update(prec1, img.size(0))
        top3.update(prec3, img.size(0))

        print(prec1)

        # compute gradient and do SGD step
        img_optimizer.zero_grad()
        diff_optimizer.zero_grad()
        loss.backward()
        img_optimizer.step()
        diff_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



def main(args):
    # cudnn.benchmark = True
    dataset, num_classes, train_loader, val_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.batch_size, args.workers)
    print(num_classes)

    # Create model
    feature_extractor = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, cut_at_pooling=True)
    img_high_level = models.create(args.arch, num_features=512,
                          dropout=args.dropout, num_classes=num_classes)
    diff_high_level = models.create(args.arch, num_features=512,
                          dropout=args.dropout, num_classes=num_classes)
    # model = nn.DataParallel(model).cuda()

    start_epoch = best_top1 = 0
    # Criterion
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    img_optimizer = torch.optim.SGD(img_high_level.classifier.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    diff_optimizer = torch.optim.SGD(diff_high_level.classifier.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
        lr = args.lr * (0.1 ** (epoch // 150))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # param_group['lr'] = param_group['lr']/2


    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(img_optimizer, epoch)
        adjust_learning_rate(diff_optimizer, epoch)

        # train for one epoch
        train(train_loader, feature_extractor, img_high_level, diff_high_level,criterion ,img_optimizer, diff_optimizer, epoch)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DAIN")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='GTOS_256')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=50)
    parser.add_argument('-j', '--workers', type=int, default=4)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet18',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)

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
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    main(parser.parse_args())


