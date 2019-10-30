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
from trainers_var_single import Trainer
from evaluators_single_view import Evaluator
import sys
from utils.logging import Logger

def get_data(name, split_id, data_dir, height, width, batch_size, workers, combine_trainval):
    root = osp.join(data_dir, name)
    dataset = Dataset(root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_class



    if name == "GTOS_256":
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

    if name == "CDMS_174":
        train_transformer_img = T.Compose([
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # normalizer,
        ])

        test_transformer_img = T.Compose([
            T.CenterCrop(256),
            T.ToTensor(),
            # normalizer,
        ])


    train_loader = DataLoader(
        Preprocessor(dataset.train, root=dataset.images_dir, dataset_name = name,
                     transform_img=train_transformer_img),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)


    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir, dataset_name = name,
                     transform_img=test_transformer_img),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(dataset.test, root=dataset.images_dir, dataset_name = name,
                     transform_img=test_transformer_img),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader



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
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.combine_trainval)


    # Create model

    img_branch_base = models.create(args.arch, cut_layer=args.cut_layer, num_classes = num_classes, num_features=args.features)
    checkpoint = load_checkpoint(args.base)
    img_branch_base.load_state_dict(checkpoint['state_dict_img'])

    img_branch = models.create('ResNet_var', img_branch_base, cut_layer=args.cut_layer, num_classes = num_classes,num_features=args.features)
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

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Evaluator
    evaluator = Evaluator(img_branch, criterion)
    if args.evaluate:
        # print("Validation:")
        # top1, _ = evaluator.evaluate(val_loader)
        # print("Validation acc: {:.1%}".format(top1))
        print("Test:")
        top1, (gt, pred) = evaluator.evaluate(test_loader)
        print("Test acc: {:.1%}".format(top1))
        from confusion_matrix import plot_confusion_matrix
        plot_confusion_matrix(gt, pred, dataset.classes, args.logs_dir)
        return


    img_param_groups = [
        # {'params': img_branch.module.base_model.base.layer4.parameters(), 'lr_mult': 1},
        {'params': img_branch.module.mean_fc.parameters(), 'lr_mult': 1},
        {'params': img_branch.module.var_fc.parameters(), 'lr_mult': 1},
        {'params': img_branch.module.classifier.parameters(), 'lr_mult': 1},
    ]


    img_optimizer = torch.optim.SGD(img_param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(img_branch, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        lr = args.lr * (1 ** (epoch // step_size))
        for g in img_optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, img_optimizer)
        if epoch < args.start_save:
            continue
        top1, _ = evaluator.evaluate(val_loader)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict_img': img_branch.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    img_branch.module.load_state_dict(checkpoint['state_dict_img'])
    top1, (gt, pred) = evaluator.evaluate(test_loader)
    from confusion_matrix import plot_confusion_matrix
    plot_confusion_matrix(gt, pred, dataset.classes, args.logs_dir)
    print('\n * Test Accuarcy: {:5.1%}\n'.format(top1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DAIN")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='CDMS_174')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', type=bool, default=False)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--cut-layer', type=str, default='layer2')
    parser.add_argument('--base', type=str, default='', metavar='PATH')
    # optimizer
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=30)
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

