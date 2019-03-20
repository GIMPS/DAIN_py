from evaluation_metrics import accuracy
from utils.meters import AverageMeter
import time

class Evaluator(object):
    def __init__(self, img_model, diff_model, depth_model, criterion):
        super(Evaluator, self).__init__()
        self.img_model = img_model
        self.diff_model = diff_model
        self.depth_model = depth_model
        self.criterion = criterion

    def evaluate(self, data_loader, print_freq = 1):
        self.img_model.eval()
        self.diff_model.eval()
        self.depth_model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1, prec3 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            top1.update(prec1, targets.size(0))
            top3.update(prec3, targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    i+1, len(data_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))
        return top1.avg

    def _parse_data(self, inputs):
        img, diff, depth, target = inputs
        img = img.float()
        diff = diff.float()
        depth = depth.float()
        target = target.cuda()
        return (img, diff, depth), target

    def _forward(self,inputs, targets):
        img, diff, depth = inputs
        img_feature_map, img_vector, _ = self.img_model(img)
        depth_feature_map, depth_vector,_ = self.depth_model(depth, img_feature_map, img_vector)
        _, _, outputs = self.diff_model(diff, depth_feature_map, depth_vector)
        loss = self.criterion(outputs, targets)
        prec1, prec3 = accuracy(outputs, targets, topk=(1,3))
        return loss, prec1, prec3


