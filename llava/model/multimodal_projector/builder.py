import torch
import torch.nn as nn
import re
from einops import rearrange
from functools import partial
from torch.nn import LayerNorm
from timm.models.regnet import RegStage
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "cabs":
        config.encoder_hidden_size = config.mm_hidden_size
        config.pos_emb = True
        config.prenorm = False
        # import os
        # if int(os.environ.get("RANK", 0)) == 0:
        #     from IPython import embed; embed()
        # from time import sleep
        # sleep(1000000)
        return CAbstractor(config, 576, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def build_pos_embeds(
    config, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.randn(1, num_input_tokens, vision_hidden_size) / (50**0.5))
        # pos_emb = torch.clamp(pos_emb, min=-2, max=2)
        # device = pos_emb.device
        # nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
        # pos_emb = pos_emb.to(torch.bfloat16).to(device)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config, output_hidden_size: int):
    # think tokens
    config.num_eos_tokens = 1
    config.initializer_range = 0.02
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size) / (50**0.5))
        # eos_tokens = torch.clamp(eos_tokens, min=-2, max=2)
        # device = eos_tokens.device
        # nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
        # eos_tokens = eos_tokens.to(torch.bfloat16).to(device)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config,
        num_input_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x


class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x


class CAbstractor(ConvProjector):
    """C-Abstractor"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
