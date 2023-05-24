import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
from engine.logger import get_logger
import numpy as np


logger = get_logger()

class CRXP(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                 norm_layer=nn.BatchNorm2d):
        super(CRXP, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        super().__init__()

        from .encoders.dual_segformer import mit_b2 as backbone
        self.rgb_backbone = backbone(norm_fuse=norm_layer)
        self.x_backbone = backbone(norm_fuse=norm_layer)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    # def initialize_parameters(self):
    #     nn.init.normal_(self.token_embedding.weight, std=0.02)
    #     nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        #
        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def build_attention_mask(self):
    #     # lazily create causal attention mask, with full attention between the vision tokens
    #     # pytorch uses additive attention mask; fill with -inf
    #     mask = torch.empty(self.context_length, self.context_length)
    #     mask.fill_(float("-inf"))
    #     mask.triu_(1)  # zero out the lower diagonal
    #     return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_rgb(self, rgb):
        return self.visual(rgb.type(self.dtype))

    def encode_x(self, x):
        return self.visual(x.type(self.dtype))


    def forward(self, rgb, x):
        rgb_features = self.encode_rgb(rgb)
        x_features = self.encode_x(x)

        # normalized features
        rgb_features = rgb_features / rgb_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * rgb_features @ x_features.t()
        logits_per_x = logits_per_image.t()

        return logits_per_image, logits_per_x