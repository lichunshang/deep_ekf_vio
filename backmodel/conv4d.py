import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class custom4D(nn.Module):
    def __init__(self, fdima, fdimb, withbn=True, full=True,groups=1) -> None:
        super().__init__()
        self.proj = nn.Sequential(projfeat4d(fdima, 2, 1, with_bn=withbn,groups=groups),
                                  nn.ReLU(inplace=True),)
        self.conva1 = sepConv4dBlock(2,4,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.conva2 = sepConv4dBlock(4,4,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb3 = sepConv4dBlock(4,8,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.convb2 = sepConv4dBlock(8,8,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb1 = sepConv4dBlock(8,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
    def forward(self,x):
        out = self.proj(x)
        out1 = self.conva1(out)
        out2 = self.conva2(out1)
        out2 = out1 + out2
        out3 = self.convb3(out2)
        out1 = self.convb2(out3)
        out1 = out3 + out1
        out = self.convb1(out1)
        out = out + out1
        return out
        


class butterfly4D(torch.nn.Module):
    '''
    butterfly 4d
    '''
    def __init__(self, fdima, fdimb, withbn=True, full=True,groups=1):
        super(butterfly4D, self).__init__()
        self.proj = nn.Sequential(projfeat4d(fdima, fdimb, 1, with_bn=withbn,groups=groups),
                                  nn.ReLU(inplace=True),)
        self.conva1 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.conva2 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.convb3 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb2 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb1 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)

    #@profile
    def forward(self,x):
        out = self.proj(x)
        b,c,u,v,h,w = out.shape # 9x9

        out1 = self.conva1(out) # 5x5, 3
        _,c1,u1,v1,h1,w1 = out1.shape

        out2 = self.conva2(out1) # 3x3, 9
        _,c2,u2,v2,h2,w2 = out2.shape

        out2 = self.convb3(out2) # 3x3, 9

        tout1 = F.upsample(out2.view(b,c,u2,v2,-1),(u1,v1,h2*w2),mode='trilinear').view(b,c,u1,v1,h2,w2) # 5x5
        tout1 = F.upsample(tout1.view(b,c,-1,h2,w2),(u1*v1,h1,w1),mode='trilinear').view(b,c,u1,v1,h1,w1) # 5x5
        out1 = tout1 + out1
        out1 = self.convb2(out1)

        tout = F.upsample(out1.view(b,c,u1,v1,-1),(u,v,h1*w1),mode='trilinear').view(b,c,u,v,h1,w1)
        tout = F.upsample(tout.view(b,c,-1,h1,w1),(u*v,h,w),mode='trilinear').view(b,c,u,v,h,w)
        out = tout + out
        out = self.convb1(out)

        return out
    
class projfeat4d(torch.nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride, with_bn=True,groups=1):
        super(projfeat4d, self).__init__()
        self.with_bn = with_bn
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, 1, (stride,stride,1), padding=0,bias=not with_bn,groups=groups)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self,x):
        b,c,u,v,h,w = x.size()
        x = self.conv1(x.view(b,c,u,v,h*w))
        if self.with_bn:
            x = self.bn(x)
        _,c,u,v,_ = x.shape
        x = x.view(b,c,u,v,h,w)
        return x
class sepConv4d(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 3D convolutions
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1), with_bn=True, ksize=3, full=True,groups=1):
        super(sepConv4d, self).__init__()
        bias = not with_bn
        self.isproj = False
        self.stride = stride[0]
        expand = 1

        if with_bn:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0,groups=groups),
                                          nn.BatchNorm2d(out_planes))
            if full:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=(1,self.stride,self.stride), bias=bias, padding=(0,ksize//2,ksize//2),groups=groups),
                                           nn.BatchNorm3d(in_planes))
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=1,                           bias=bias, padding=(0,ksize//2,ksize//2),groups=groups),
                                           nn.BatchNorm3d(in_planes))
            self.conv2 = nn.Sequential(nn.Conv3d(in_planes, in_planes*expand, (ksize,ksize,1), stride=(self.stride,self.stride,1), bias=bias, padding=(ksize//2,ksize//2,0),groups=groups),
                                       nn.BatchNorm3d(in_planes*expand))
        else:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0,groups=groups)
            if full:
                self.conv1 = nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=(1,self.stride,self.stride), bias=bias, padding=(0,ksize//2,ksize//2),groups=groups)
            else:
                self.conv1 = nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=1,                           bias=bias, padding=(0,ksize//2,ksize//2),groups=groups)
            self.conv2 = nn.Conv3d(in_planes, in_planes*expand, (ksize,ksize,1), stride=(self.stride,self.stride,1), bias=bias, padding=(ksize//2,ksize//2,0),groups=groups)
        self.relu = nn.ReLU(inplace=True)
        
    #@profile
    def forward(self,x):
        b,c,u,v,h,w = x.shape
        x = self.conv2(x.view(b,c,u,v,-1)) # WTA convolution over (u,v)
        b,c,u,v,_ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b,c,-1,h,w)) # spatial convolution over (x,y)
        b,c,_,h,w = x.shape

        if self.isproj:
            x = self.proj(x.view(b,c,-1,w))
        x = x.view(b,-1,u,v,h,w)
        return x


class sepConv4dBlock(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 2D convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1), with_bn=True, full=True,groups=1):
        super(sepConv4dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            if full:
                self.downsample = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn,ksize=1, full=full,groups=groups)
            else:
                self.downsample = projfeat4d(in_planes, out_planes,stride[0], with_bn=with_bn,groups=groups)
        self.conv1 = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, full=full ,groups=groups)
        self.conv2 = sepConv4d(out_planes, out_planes,(1,1,1), with_bn=with_bn, full=full,groups=groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    #@profile
    def forward(self,x):
        out = self.relu1(self.conv1(x))
        if self.downsample:
            x = self.downsample(x)
        out = self.relu2(x + self.conv2(out))
        return out

class flow_reg(nn.Module):
    """
    Soft winner-take-all that selects the most likely diplacement.
    Set ent=True to enable entropy output.
    Set maxdisp to adjust maximum allowed displacement towards one side.
        maxdisp=4 searches for a 9x9 region.
    Set fac to squeeze search window.
        maxdisp=4 and fac=2 gives search window of 9x5
    """
    def __init__(self, size, ent=False, maxdisp = int(4), fac=1):
        B,W,H = size
        super(flow_reg, self).__init__()
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3  # by default using truncation 7x7

        flowrangey = range(-maxdisp,maxdisp+1)
        flowrangex = range(-int(maxdisp//self.fac),int(maxdisp//self.fac)+1)
        meshgrid = np.meshgrid(flowrangex,flowrangey)
        flowy = np.tile( np.reshape(meshgrid[0],[1,2*maxdisp+1,2*int(maxdisp//self.fac)+1,1,1]), (B,1,1,H,W) )
        flowx = np.tile( np.reshape(meshgrid[1],[1,2*maxdisp+1,2*int(maxdisp//self.fac)+1,1,1]), (B,1,1,H,W) )
        self.register_buffer('flowx',torch.Tensor(flowx))
        self.register_buffer('flowy',torch.Tensor(flowy))

        self.pool3d = nn.MaxPool3d((self.wsize*2+1,self.wsize*2+1,1),stride=1,padding=(self.wsize,self.wsize,0))

    def forward(self, x):
        b,u,v,h,w = x.shape
        oldx = x

        if self.truncated:
            # truncated softmax
            x = x.view(b,u*v,h,w)

            idx = x.argmax(1)[:,np.newaxis]
            if x.is_cuda:
                mask = Variable(torch.cuda.HalfTensor(b,u*v,h,w)).fill_(0)
            else:
                mask = Variable(torch.FloatTensor(b,u*v,h,w)).fill_(0)
            mask.scatter_(1,idx,1)
            mask = mask.view(b,1,u,v,-1)
            mask = self.pool3d(mask)[:,0].view(b,u,v,h,w)

            ninf = x.clone().fill_(-np.inf).view(b,u,v,h,w)
            x = torch.where(mask.byte(),oldx,ninf)
        else:
            self.wsize = (np.sqrt(u*v)-1)/2

        b,u,v,h,w = x.shape
        x = F.softmax(x.view(b,-1,h,w),1).view(b,u,v,h,w)
        outx = torch.sum(torch.sum(x*self.flowx,1),1,keepdim=True)
        outy = torch.sum(torch.sum(x*self.flowy,1),1,keepdim=True)

        if self.ent:
            # local
            local_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.
            else:
                local_entropy /= np.log((self.wsize*2+1)**2)

            # global
            x = F.softmax(oldx.view(b,-1,h,w),1).view(b,u,v,h,w)
            global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            global_entropy /= np.log(x.shape[1]*x.shape[2])
            return torch.cat([outx,outy],1),torch.cat([local_entropy, global_entropy],1)
        else:
            return torch.cat([outx,outy],1),None

if __name__ == '__main__':
    a = torch.rand(32,1,12,40,12,40)
    conv = custom4D(1,8)
    # f = flow_reg()
    b = conv(a)
    print(b.shape)