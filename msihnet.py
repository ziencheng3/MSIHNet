import torch.optim
import torch.nn as nn
from invblock import INV_block,INV_block_final

import Unet_common as common

dwt = common.DWT()
iwt = common.IWT()

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

class SIH(nn.Module):
    def __init__(self):
        super(SIH, self).__init__()
        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

        self.inv9 = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()
        self.inv13 = INV_block()
        self.inv14 = INV_block()
        self.inv15 = INV_block()
        self.inv16 = INV_block_final()

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

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out,t1 = self.inv16(out)
            return out,t1
        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
            return out

class MSIHNet(nn.Module):
    def __init__(self):
        super(MSIHNet, self).__init__()
        self.sih1 = SIH()
        self.sih2 = SIH()
        self.sih3 = SIH()
        self.sih4 = SIH()

    def forward(self, x1,x2, rev=False):
        if not rev:
            LL = x2.narrow(1, 0, 3)
            HL = x2.narrow(1, 3, 3)
            LH = x2.narrow(1, 6, 3)
            HH = x2.narrow(1, 9, 3)
            LL_input = dwt(LL)
            HL_input = dwt(HL)
            LH_input = dwt(LH)
            HH_input = dwt(HH)
            in1 = torch.cat((LH_input, HH_input), 1)
            out1,t1 = self.sih1(in1)
            LH1_input = out1.narrow(1,0,12)
            in2 = torch.cat((LH1_input, HL_input), 1)
            out2,t2 = self.sih2(in2)
            LH2_input = out2.narrow(1,0,12)
            in3 = torch.cat((x1, LH2_input), 1)
            out3,t3 = self.sih3(in3)
            cover1_input = out3.narrow(1,0,12)
            in4 = torch.cat((cover1_input,LL_input), 1)
            out4, t4 = self.sih4(in4)
            t = torch.cat((t1, t2, t3, t4), 1)
            return out4, t
        else:
            t1 = x2.narrow(1, 0, 12)
            t2 = x2.narrow(1, 12, 12)
            t3 = x2.narrow(1, 24, 12)
            t4 = x2.narrow(1, 36, 12)
            in4 = torch.cat((x1, t4), 1)
            out4 = self.sih4(in4,rev=True)
            LL_input = out4.narrow(1,12,12)
            cover1_input = out4.narrow(1,0,12)
            in3 = torch.cat((cover1_input, t3), 1)
            out3 = self.sih3(in3,rev=True)
            LH2_input = out3.narrow(1,12,12)
            in2 = torch.cat((LH2_input, t2), 1)
            out2 = self.sih2(in2,rev=True)
            HL_input = out2.narrow(1,12,12)
            LH1_input = out2.narrow(1,0,12)
            in1 = torch.cat((LH1_input, t1), 1)
            out1 = self.sih1(in1,rev=True)
            HH_input = out1.narrow(1, 12, 12)
            LH_input = out1.narrow(1, 0, 12)
            LL = iwt(LL_input)
            HL = iwt(HL_input)
            LH = iwt(LH_input)
            HH = iwt(HH_input)
            secret = torch.cat((LL, HL, LH, HH), 1)
            return secret



















