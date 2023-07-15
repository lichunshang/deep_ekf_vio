import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backmodel.core.update import BasicUpdateBlock, SmallUpdateBlock
from backmodel.core.extractor import BasicEncoder, SmallEncoder
from backmodel.core.corr import CorrBlock, AlternateCorrBlock
from backmodel.core.utils.utils import bilinear_sampler, coords_grid, upflow8
from backmodel.core.update import GMAUpdateBlock
from backmodel.core.gma import Attention, Aggregate

class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
    
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        corr_levels = 4
        corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0)
        self.update_block = GMAUpdateBlock( hidden_dim=hdim)
        self.att = Attention(dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
  
    def prepare(self, image1, image2):
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=4)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        attention = self.att(inp)
        return corr_fn, net, inp, attention
    
    def update(self, coords0, coords1, corr_fn, net, inp, attention):
        coords1 = coords1.detach()
        corr = corr_fn(coords1) # index correlation volume

        flow = coords1 - coords0
        net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)
        
        # F(t+1) = F(t) + \Delta(t)
        coords1 = coords1 + delta_flow
        flow_up = self.upsample_flow(coords1 - coords0,up_mask)
        return coords1, net, flow_up

    def forward(self, image1, image2, iters=12):
        """ Estimate optical flow between pair of frames """

        coords0, coords1 = self.initialize_flow(image1)
        corr_fn, net, inp, attn = self.prepare(image1, image2)
        list_pred = []
        for itr in range(iters):
            coords1, net, flow = self.update(coords0, coords1, corr_fn, net, inp, attn)
            list_pred.append(flow)
        return list_pred
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        # image1 = image1.contiguous()
        # image2 = image2.contiguous()

        # hdim = self.hidden_dim
        # cdim = self.context_dim

        # # run the feature network
        # fmap1, fmap2 = self.fnet([image1, image2])        
        
        # fmap1 = fmap1.float()
        # fmap2 = fmap2.float()
        # corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # # run the context network
        # cnet = self.cnet(image1)
        # net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        # net = torch.tanh(net)
        # inp = torch.relu(inp)

        # coords0, coords1 = self.initialize_flow(image1)

        # if flow_init is not None:
        #     coords1 = coords1 + flow_init

        # flow_predictions = []
        # for itr in range(iters):
        #     coords1 = coords1.detach()
        #     corr = corr_fn(coords1) # index correlation volume

        #     flow = coords1 - coords0
        #     net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

        #     # F(t+1) = F(t) + \Delta(t)
        #     coords1 = coords1 + delta_flow

        #     # upsample predictions
        #     if up_mask is None:
        #         flow_up = upflow8(coords1 - coords0)
        #     else:
        #         flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
        #     flow_predictions.append(flow_up)

        # if test_mode:
        #     return coords1 - coords0, flow_up
            
        # return flow_predictions

if __name__ == '__main__':
    a = torch.rand(32,3,128,320).cuda()
    b = torch.rand(32,3,128,320).cuda()
    model = RAFT().cuda()
    out = model(a,b)
    for i,f in enumerate(out):
        print(i)
        print(f.shape)