from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_layer='layer3', num_classes = 0):
        super(ResNet, self).__init__()

        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        out_planes = self.base.fc.in_features

        self.num_classes = num_classes
        # Change the num_features to CNN output channels
        self.num_features = out_planes

        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

        self.low_level_modules = nn.ModuleList([])
        self.high_level_modules = nn.ModuleList([])

        next_module_belong = 'low_level'
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            if next_module_belong == 'low_level':
                self.low_level_modules.append(module)
            else:
                self.high_level_modules.append(module)
            if name == cut_layer:
                next_module_belong = 'high_level'

    def forward(self, x, fusion_feature=None, fusion_fc=None):
        for _, layer in enumerate(self.low_level_modules):
            x = layer(x)
        feature_map = x
        if fusion_feature is not None:
            x = x + fusion_feature
        for _, layer in enumerate(self.high_level_modules):
            x = layer(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if fusion_fc is not None:
            x = (x + fusion_fc) / 2
        return feature_map, x


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
