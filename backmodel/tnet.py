import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from params import par

class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)

class TNet(nn.Module):
    def __init__(self, pretrained_path='pretrained/efficientnet-b0.pth'):
        super().__init__()
        
        config = {
            "backbone": pretrained_path,
            "learn_embedding_with_pose_token": False,
            "num_t_encoder_layers": par.num_t_encoder_layers,
            "num_t_decoder_layers": par.num_t_decoder_layers,
            "num_rot_encoder_layers": par.num_rot_encoder_layers,
            "num_rot_decoder_layers": par.num_rot_decoder_layers,
            "hidden_dim": par.hidden_dim,
            "reduction": par.reduction,
            "dim_feedforward": par.dim_feedforward,
        }
        
        config_t = {**config}
        config_rot = {**config}
        self.backbone = build_backbone(config)

        self.transformer_t = Transformer(config_t)
        self.transformer_rot = Transformer(config_rot)
        decoder_dim = self.transformer_t.d_model
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)
        self.query_embed_t = nn.Embedding(1, decoder_dim)
        self.query_embed_rot = nn.Embedding(1, decoder_dim)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 3)

    def forward_transformers(self, data):
        if isinstance(data, (list, torch.Tensor)):
            data = nested_tensor_from_tensor_list(data)
        features, pos = self.backbone(data)
        src_t, mask_t = features[0].decompose()
        src_rot, mask_rot = features[1].decompose()

        # assert mask_t is not None
        # assert mask_rot is not None

        desc_t = self.transformer_t(self.input_proj_t(src_t), mask_t, self.query_embed_t.weight, pos[0])[0][0]
        desc_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, self.query_embed_rot.weight, pos[1])[0][0]

        return desc_t, desc_rot
    def forward_heads(self, transformer_res):
        desc_t, desc_rot = transformer_res
        x_t = self.regressor_head_t(desc_t)
        x_rot = self.regressor_head_rot(desc_rot)
        output = torch.cat((x_t,x_rot), dim=1)
        return output
    
    def forward(self, data):
        res = self.forward_transformers(data)
        if not par.use_lstm:
            res = self.forward_heads(res)
        return res

if __name__ =='__main__':
    config = {
        "num_t_encoder_layers": par.num_t_encoder_layers,
        "num_t_decoder_layers": par.num_t_decoder_layers,
        "num_rot_encoder_layers": par.num_rot_encoder_layers,
        "num_rot_decoder_layers": par.num_rot_decoder_layers,
        "hidden_dim": par.hidden_dim,
        "reduction": par.reduction,
        "dim_feedforward": par.dim_feedforward
    }
    model = TNet(config)
    print(model)
        