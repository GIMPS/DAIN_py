from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from utils import to_torch
import torch.nn as nn
from torchvision import models

original_model = models.vgg16(pretrained=True)
print(list(original_model.features.children()))
class vggConv4(nn.Module):
            def __init__(self):
                super(vggConv4, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-3]
                )
            def forward(self, x):
                x = self.features(x)
                return x


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
