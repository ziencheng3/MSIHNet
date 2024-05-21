import torch
import torch.nn as nn
import config as c
from modules import InvertibleConv1x1
from rrdb_denselayer import ResidualDenseBlock_out
from ActNorm import ActNorm




class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=9):
        super().__init__()
        if harr:
            self.split_len1 = in_1
            self.split_len2 = in_2
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

        in_channels = 12
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
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
            out = torch.cat((y1, y2), 1)


        else:  # names of x and y are swapped!
            x1, x2 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2))

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)

            x = torch.cat((y1, y2), 1)
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)
            out = self.actnorm.reverse(out)

        return out







class pre_Hinet(nn.Module):

    def __init__(self):
        super(pre_Hinet, self).__init__()

        self.inv1 = INV_block_affine()
        self.inv2 = INV_block_affine()
        self.inv3 = INV_block_affine()
        self.inv4 = INV_block_affine()
        self.inv5 = INV_block_affine()
        self.inv6 = INV_block_affine()
        self.inv7 = INV_block_affine()
        self.inv8 = INV_block_affine()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
            return out
        else:

            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
            return out

if __name__ == '__main__':
    model = pre_Hinet()
    image = torch.rand((1, 12, 64, 64))
    out = model(image,rev=True)
    print(out.shape)