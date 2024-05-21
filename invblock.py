from math import exp
import torch
import torch.nn as nn
import config as c
from rrdb_denselayer import ResidualDenseBlock_out
from ActNorm import ActNorm
from modules import InvertibleConv1x1



class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        in_channels = 24
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        self.actnorm = ActNorm(in_channels)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1),
        #           x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x = self.actnorm(x)
            x, logdet = self.flow_permutation(x, logdet=0, rev=False)
            x1, x2 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2))


            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
            out = torch.cat((y1, y2), 1)

        else:
            x1, x2 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2))


            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

            x = torch.cat((y1, y2), 1)
            # inv permutation
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)
            out = self.actnorm.reverse(out)
        return out



class INV_block_final(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        in_channels = 24
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        self.actnorm = ActNorm(in_channels)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        if not rev:
            x = self.actnorm(x)
            x, logdet = self.flow_permutation(x, logdet=0, rev=False)
            x1, x2 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2))


            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
            out = torch.cat((y1, y2), 1)
            #t1是因为第二次隐藏后用t1作为反向的输入，x2是第二次隐藏要隐藏的图片
            return out, t1

        else:
            x1, x2 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2))


            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

            x = torch.cat((y1, y2), 1)
            # inv permutation
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)
            out = self.actnorm.reverse(out)
            return out
