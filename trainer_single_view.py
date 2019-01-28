from __future__ import print_function, absolute_import
import time

from evaluation_metrics import accuracy
from utils.meters import AverageMeter




class Trainer(object):
    def __init__(self, img_model, criterion):
        super(Trainer, self).__init__()
        self.img_model = img_model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer1, print_freq=1):
        self.img_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        img, target = inputs
        img = img.float()
        target = target.cuda()
        return img, target

    def _forward(self,inputs, targets):
        img = inputs
        img_feature_map, outputs = self.img_model(img)
        loss = self.criterion(outputs, targets)
        prec, = accuracy(outputs, targets)
        return loss, prec

