from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import numpy as np

__all__ = ['ResNet_var']


class ResNet_var(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, base_model,  pretrained=True, cut_layer='layer3', num_classes=0, num_features=0):
        super(ResNet_var, self).__init__()

        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        # self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.base_model = base_model
        self.base = self.base_model.base

        out_planes = self.base.fc.in_features

        self.num_classes = num_classes

        self.num_features = num_features
        self.has_embedding = num_features > 0

        # # Append new layers
        # if self.has_embedding:
        #     self.feat = nn.Linear(out_planes, self.num_features)
        #     self.feat_bn = nn.BatchNorm1d(self.num_features)
        #     init.kaiming_normal_(self.feat.weight, mode='fan_out')
        #     init.constant_(self.feat.bias, 0)
        #     init.constant_(self.feat_bn.weight, 1)
        #     init.constant_(self.feat_bn.bias, 0)
        # else:
        #     # Change the num_features to CNN output channels
        #     self.num_features = out_planes

        self.num_features = out_planes

        # self.mean_fc = self.base_model.feat
        # self.mean_fc = nn.Linear(out_planes, num_features)
        self.var_fc = nn.Linear(out_planes, num_features)

        # init.normal_(self.mean_fc.weight, std=0.001)
        # init.constant_(self.mean_fc.bias, 0)

        # init.constant_(self.var_fc.weight, 1)
        # init.constant_(self.var_fc.bias, 0)

        init.normal_(self.var_fc.weight, std=0.001)
        init.constant_(self.var_fc.bias, 0)
        self.classifier = self.base_model.classifier

        # if self.num_classes > 0:
        #     self.classifier = nn.Linear(self.num_features, self.num_classes)
        #     init.normal_(self.classifier.weight, std=0.001)
        #     init.constant_(self.classifier.bias, 0)

        # self.low_level_modules = nn.ModuleList([])
        # self.high_level_modules = nn.ModuleList([])

        # next_module_belong = 'low_level'
        # for name, module in self.base._modules.items():
        #     if name == 'avgpool':
        #         break
        #     if next_module_belong == 'low_level':
        #         self.low_level_modules.append(module)
        #     else:
        #         self.high_level_modules.append(module)
        #     if name == cut_layer:
        #         next_module_belong = 'high_level'

    def forward(self, x, fusion_feature=None, fusion_vector=None):
        for _, layer in enumerate(self.base_model.low_level_modules):
            x = layer(x)
        feature_map = x
        if fusion_feature is not None:
            x = x + fusion_feature
        for _, layer in enumerate(self.base_model.high_level_modules):
            x = layer(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        # if self.has_embedding:
        #     x = self.feat(x)
        #     x = self.feat_bn(x)
        #     x = F.relu(x)
        #
        feature_vector = x
        #
        # if fusion_vector is not None:
        #     x = x + fusion_vector

        if self.training is False:
            # x = self.mean_fc(x)
            # x = F.relu(x)
            pass
        else:
            mean_fv = x
            # mean_fv = self.mean_fc(x)
            var_fv = self.var_fc(x)
            s = np.random.normal(0, 1)
            x = mean_fv + s * var_fv
            # x = F.relu(x)

        x = self.classifier(x)
        return feature_map, feature_vector, x
