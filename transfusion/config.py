from dataclasses import dataclass, field, MISSING
from typing import List, Tuple


def fix(blah): return field(default_factory=lambda: blah)


@dataclass
class ModelConfig:
    # general params
    vocab_size: int = 29

    # main transformer params
    layers: int = 24
    dim: int = 768
    nheads: int = 8
    dropout: float = 0.1
    attention_type: str = 'wavlm' # either 'normal' or 'wavlm'
    # wavlm attention paramters
    wavlm_num_bucket: int = 140
    wavlm_max_dist: int = 280

    # timestep embedding params
    t_emb_dim: int = 768
    t_emb_max_period: int = 10000
    T: int = 200

    # conditioning params
    cond_emb_dim: int = 1024
    drop_cond_prob: float = 0.1 # for classifier free guidance (same as original paper)
    # For 12 layers: fix([0, 3, 6, 9]) | For 24 layers: fix([0, 4, 8, 12, 16, 20])
    cond_cross_attn_layers: List[int] = fix([0, 4, 8, 12, 16, 20])
    # relative positional encoding params
    conv_pos: int = 256 # typically 128, here 256
    conv_pos_groups: int = 32 # typically 16, here 32

    diffusion_type: str = 'multinomial' # only 'multinomial' supported
    diffusion_s: float = 0.008

    # transformer positional encoding
    # either 'relative' or 'absolute', our models use relative
    # to allow arbitrary lengths at inference time.
    pos_encoding: str = 'relative' 
    
    # waveform encoder
    wav_encoder: str = 'wavlm' # only 'wavlm' currently supported


