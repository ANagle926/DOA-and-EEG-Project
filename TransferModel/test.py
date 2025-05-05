import torch
import torch.nn as nn

from EEGPT.pretrain.modeling_pretraining import EEGPT

model = EEGPT(
    num_electrodes=58,
    patch_size=64,
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4
)
