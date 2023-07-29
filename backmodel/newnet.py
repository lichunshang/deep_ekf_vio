import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par
import torchvision.models as models
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_b, Swin_V2_B_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)
def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )
def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        # x = self.attn(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out = self.sa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Reg(nn.Module):
    def __init__(self, inputnum=2) -> None:
        super().__init__()
        self.inputnum = inputnum

        self.inplanes = 64
        self.deconv_with_bias = False
        layers = [2,2,2,2]
        # layers = [3,4,6,3]

        self.conv1 = nn.Conv2d(inputnum, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(num_layers=3, num_filters=(256,256,256), num_kernels=(4,4,4))
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=12,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.final_pooling = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            # layers.append(ChannelAttention(planes))
            # layers.append(SpatialAttention())
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes


        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x = self.final_pooling(x)
        # print(x.shape)
        return x.squeeze()

class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        # self.model = raft_small(weights = Raft_Small_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, x):
        x1 = x[:,0:3,:]
        x2 = x[:,3:6,:]
        res = self.model(x1,x2)
        return res



class BasicBlock1(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock1, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)



class Res(nn.Module):
    def __init__(self, inputnum=2):
        super(Res, self).__init__()

        blocknums = [3,4,6,7,3]
        outputnums = [64,128,128,256,256]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32

        self.layer1 = self._make_layer(BasicBlock1, outputnums[0], blocknums[0], 2, 1, 1) # 40 x 28
        self.layer2 = self._make_layer(BasicBlock1, outputnums[1], blocknums[1], 2, 1, 1) # 20 x 14
        self.layer3 = self._make_layer(BasicBlock1, outputnums[2], blocknums[2], 2, 1, 1) # 10 x 7
        self.layer4 = self._make_layer(BasicBlock1, outputnums[3], blocknums[3], 2, 1, 1) # 5 x 4
        self.layer5 = self._make_layer(BasicBlock1, outputnums[4], blocknums[4], 2, 1, 1) # 3 x 2
        
        
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class PoseRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        fcnum = 256 * 10

        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)
        
        fc1_covar = linear(fcnum, 128)
        fc2_covar = linear(128,32)
        fc3_covar = nn.Linear(32,6)

        self.trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        # for param in self.trans.parameters():
        #     param.requires_grad = False
        self.rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        # for param in self.rot.parameters():
        #     param.requires_grad = False
        self.covar = nn.Sequential(fc1_covar, fc2_covar, fc3_covar)
        # for param in self.covar.parameters():
        #     param.requires_grad = False
    def forward(self,x):

        x = x.view(x.shape[0], -1)
        trans = self.trans(x)
        rot = self.rot(x)
        covar = self.covar(x)
        x = torch.cat((rot, trans, covar), dim=1)
        return x


if __name__ == '__main__':
    import torchvision
    img1= '/mnt/data/teamAI/duy/data/kitti/2011_10_03/2011_10_03_drive_0034_extract/image_02/data/0000000000.png'
    img2 = '/mnt/data/teamAI/duy/data/kitti/2011_10_03/2011_10_03_drive_0034_extract/image_02/data/0000000001.png'
    model = RAFT()
    i1 = torchvision.io.read_image(img1).to(float)
    i2 = torchvision.io.read_image(img2).to(float)
    output = model(torch.cat((i1,i2),dim=0).unsqueeze(0))
    print(len(output))