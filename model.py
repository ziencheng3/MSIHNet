import torch.optim
import torch.nn as nn
import config as c
from msihnet import MSIHNet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = MSIHNet()

    def forward(self, x1, x2, rev=False):

        if not rev:
            out, t = self.model(x1, x2)
            return out,t

        else:
            out = self.model(x1, x2, rev=True)
            return out




def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)


