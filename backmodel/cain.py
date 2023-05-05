import math
import numpy as np

import torch
import torch.nn as nn

from .common import *
from params import par

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(3, 12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)
        feats = self.interpolate(feats1, feats2)

        return feats

class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3, in_channels=3):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=3, depth=depth)
        # self.decoder = Decoder(depth=depth)
        fcnum = int(2*in_channels*par.img_h*par.img_w)
        fc1_trans = linear(fcnum, 512)
        fc1x_trans = linear(512,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,12)

        fc1_rot = linear(fcnum, 512)
        fc1x_rot = linear(512,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        fc1_covar = linear(fcnum, 512)
        fc1x_covar = linear(512,128)
        fc2_covar = linear(128,32)
        fc3_covar = linear(32,6)

        self.trans = nn.Sequential(fc1_trans, fc1x_trans, fc2_trans, fc3_trans)
        self.rot = nn.Sequential(fc1_rot, fc1x_rot, fc2_rot, fc3_rot)
        self.covar = nn.Sequential(fc1_covar, fc1x_covar, fc2_covar,fc3_covar)

    def forward(self, x1, x2):
        # x1, m1 = sub_mean(x1)
        # x2, m2 = sub_mean(x2)

        # if not self.training:
        #     paddingInput, paddingOutput = InOutPaddings(x1)
        #     x1 = paddingInput(x1)
        #     x2 = paddingInput(x2)
        
        feats = self.encoder(x1, x2)
        feats = feats.view(feats.shape[0], -1)
        # out = self.decoder(feats)
        x_trans = self.trans(feats)
        x_rot = self.rot(feats)
        x_covar = self.covar(feats)
        # if not self.training:
        #     out = paddingOutput(out)

        # mi = (m1 + m2) / 2
        # out += mi
        out =  torch.cat((x_trans, x_rot, x_covar), dim=1)

        # return out, feats
        return out