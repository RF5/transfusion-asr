import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .wavlm_modules import MultiheadAttention


# ------------------------
# Code adapted from OpenAI guided diffusion repo

def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------
# Code adapted from fairseq hubert

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class RelativePositionalEncoder(nn.Module):

    def __init__(self, dim, conv_pos, conv_pos_groups):
        super().__init__()
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.embedding_dim = dim
        
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())

    def forward(self, x):
        """ `x` of shape (bs, seq_len, dim) """
        x_conv = self.pos_conv(x.permute(0, 2, 1)) # (bs, seq_len, dim) -> (bs, dim, seq_len)
        x_conv = x_conv.permute(0, 2, 1) # (bs, dim, seq_len) -> (bs, seq_len, dim)
        return x + x_conv


# ------------------------
# TransFusion model code

class Pogfuse(nn.Module):
    """Transformer encoder layer with cross-attention and embedding inputs"""

    def __init__(self, dim, t_emb_dim, cond_emb_dim, nheads, add_cond_seq=True,
                 layer_norm_eps: float = 1e-5, dropout=0.1, d_ff_mult=4, use_wavlm_attn=False,
                 wavlm_num_bucket=140, wavlm_max_dist=280, has_rel_attn_bias=False) -> None:
        super().__init__()
        self.t_layers = nn.Sequential(
            nn.Linear(t_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.add_cond_seq = add_cond_seq
        if add_cond_seq:
            self.cond_layers = nn.Sequential(
                nn.LayerNorm(cond_emb_dim, layer_norm_eps),
                nn.Linear(cond_emb_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
        
        self.cond_pooled_layers = nn.Sequential(
            nn.LayerNorm(cond_emb_dim, layer_norm_eps),
            nn.Linear(cond_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Attention layers
        self.activation = F.selu
        self.use_wavlm_attn = use_wavlm_attn
        if use_wavlm_attn:
            self.self_attn = MultiheadAttention(dim,
                                nheads,
                                dropout=dropout,
                                self_attention=True,
                                has_relative_attention_bias=has_rel_attn_bias,
                                num_buckets=wavlm_num_bucket,
                                max_distance=wavlm_max_dist,
                                rescale_init=False,
                                gru_rel_pos=True,
            )
        else:
            self.self_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim, dim*d_ff_mult)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim*d_ff_mult, dim)

        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, t_emb: Tensor, pooled_conv_emb: Tensor, cond_emb: Optional[Tensor] = None,
                x_padding_mask: Optional[Tensor] = None, cond_padding_mask: Optional[Tensor] = None,
                pos_bias: Optional[Tensor] = None) -> Tensor:
        """Forward with `x` (bs, seq_len, dim), summing `t_emb` (bs, dim) before the transformer layer,
        and appending `conditioning_emb` (bs, seq_len2, dim) to the key/value pairs of the attention.
        Also `pooled_conv_emb` (bs, dim) is summed with the timestep embeddings

        Optionally specify key/value padding for input `x` with `x_padding_mask` (bs, seq_len), and optionally
        specify key/value padding mask for conditional embedding with `cond_padding_mask` (bs, seq_len2).
        By default no padding is used. Good idea to use cond padding but not x padding.

        `pos_bias` is positional bias for wavlm-style attention gated relative position bias.

        Returns `x` of same shape (bs, seq_len, dim)
        """
        # -----------------------
        # 1. Get and add timestep embedding
        t = self.t_layers(t_emb)[:, None] # (bs, 1, dim)
        c_pool = self.cond_pooled_layers(pooled_conv_emb)[:, None] # (bs, 1, dim)
        x += t + c_pool # (bs, seq_len, dim)
        # -----------------------
        # 2. Get and append conditioning embeddings
        if self.add_cond_seq: c = self.cond_layers(cond_emb) # (bs, seq_len2, dim)
        else: c = None
        # -----------------------
        # 3. Do transformer layer
        x1, pos_bias = self._sa_block(x, c, x_padding_mask=x_padding_mask, c_padding_mask=cond_padding_mask, pos_bias=pos_bias)
        x = self.norm1(x + x1)
        x = self.norm2(x + self._ff_block(x))

        return x, pos_bias

    # self-attention block
    def _sa_block(self, x: Tensor, c: Optional[Tensor], c_padding_mask: Optional[Tensor],
                  x_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None,
                  pos_bias: Optional[Tensor] = None) -> Tensor:
        if x_padding_mask is None:
            x_padding_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        
        if self.add_cond_seq:
            if c_padding_mask is None:
                c_padding_mask = torch.zeros(c.shape[0], c.shape[1], dtype=torch.bool, device=c.device)

            kv = torch.concat((x, c), dim=1) # (bs, seq_len + seq_len2, dim)
            key_padding_mask = torch.concat((x_padding_mask, c_padding_mask), dim=1) # (bs, seq_len + seq_len2)
        else:
            kv = x
            key_padding_mask = x_padding_mask

        if self.use_wavlm_attn:
            # does not support batch first, so we must permute:
            # (bs, seq_len, dim) --> (seq_len, bs, dim)
            x = x.permute(1, 0, 2)
            kv = kv.permute(1, 0, 2)
            x, z, pos_bias = self.self_attn(x, kv, kv,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask, 
                                need_weights=False,
                                position_bias=pos_bias)
            x = x.permute(1, 0, 2) # swap back (seq_len, bs, dim) -> (bs, seq_len, dim)
        else:
            x = self.self_attn(x, kv, kv,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        return self.dropout1(x), pos_bias

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransFusion(nn.Module):

    def __init__(self, cfg: ModelConfig, max_transcript_len=200) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = []
        
        if cfg.pos_encoding == 'relative':
            self.pos_embedding = RelativePositionalEncoder(self.cfg.dim, self.cfg.conv_pos, self.cfg.conv_pos_groups)
        else: 
            self.pos_embedding = nn.Embedding(max_transcript_len, self.cfg.dim)
        self.conditioning_pos_emb = RelativePositionalEncoder(self.cfg.cond_emb_dim, self.cfg.conv_pos, self.cfg.conv_pos_groups)

        if self.cfg.diffusion_type == 'continuous': self.char_embedding = nn.Linear(self.cfg.vocab_size, self.cfg.dim)
        else: self.char_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
    
        for i in range(cfg.layers):
            add_cond_cross_attn = i in list(self.cfg.cond_cross_attn_layers)
            layer = Pogfuse(self.cfg.dim, self.cfg.t_emb_dim, self.cfg.cond_emb_dim,
                            self.cfg.nheads, add_cond_seq=add_cond_cross_attn, dropout=self.cfg.dropout,
                            use_wavlm_attn=cfg.attention_type == 'wavlm' and not add_cond_cross_attn, 
                            wavlm_num_bucket=cfg.wavlm_num_bucket, wavlm_max_dist=cfg.wavlm_max_dist,
                            has_rel_attn_bias=(cfg.attention_type == 'wavlm' and i == 1))
                            # add relative attn bias at i=1 as that is first attn where we do not use
                            # cross attention.
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

        if cfg.attention_type == 'wavlm':
            self.head = nn.Sequential(
                nn.LayerNorm(self.cfg.dim),
                nn.Linear(self.cfg.dim, self.cfg.vocab_size)
            )
        else:
            self.head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cfg.dim, self.cfg.vocab_size)
            )

        if cfg.wav_encoder == 'wavlm' or not hasattr(cfg, 'wav_encoder'):
            pass # we use pre-computed wavlm embeddings
        else: raise NotImplementedError()


    def forward(self, x: Tensor, t: Tensor, cond_emb: Tensor, cond_padding_mask: Tensor, 
                      x_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """ Transformer with conditioning cross attention.
        - `x`: (bs, seq_len) long tensor of character indices
            or (bs, seq_len, vocab_size) if cfg.diffusion_type == 'continuous'
        - `t`: (bs, ) long tensor of timestep indices
        - `cond_emb`: (bs, seq_len2, cond_emb_dim) if using wavlm encoder, else (bs, T)
        - `x_padding_mask`: (bs, seq_len) if using wavlm encoder, else (bs, T)
        - `cond_padding_mask`: (bs, seq_len2)

        Returns logits (bs, seq_len, vocab_size)
        """

        # 1. Base: character, timestep embeddings and zeroing
        bs = x.shape[0]
        x = self.char_embedding(x) # (bs, seq_len, dim)

        if self.cfg.pos_encoding == 'relative':
            x = self.pos_embedding(x)
        else:
            pos_emb = self.pos_embedding.weight[None].expand(bs, -1, -1) # (seq_len, dim) --> (bs, seq_len, dim)
            x = x + pos_emb

        # we use pre-computed wavlm embeddings
        if self.cfg.wav_encoder != 'wavlm': raise NotImplementedError()
            
        t_emb = timestep_embedding(t, self.cfg.t_emb_dim, self.cfg.t_emb_max_period, dtype=cond_emb.dtype) # (bs, t_dim)
        # 2. Classifier-free guidance: with prob cfg.drop_cond_prob, zero out and drop conditional probability
        if self.training:
            zero_cond_inds = torch.rand_like(t, dtype=cond_emb.dtype) < self.cfg.drop_cond_prob
        else:
            # never randomly zero when in eval mode
            zero_cond_inds = torch.zeros_like(t, dtype=torch.bool)
            if cond_padding_mask.all():
                # BUT, if all cond information is padded then we are obviously doing unconditional synthesis,
                # so, force zero_cond_inds to be all ones
                zero_cond_inds = ~zero_cond_inds

        # set mask for these conditional entries to true everywhere (i.e. mask them out)
        pooled_cond_emb = cond_emb.mean(dim=1)

        cond_emb = self.conditioning_pos_emb(cond_emb)

        if cond_padding_mask.all() == False:
            denoms = ((~cond_padding_mask).sum(dim=1)[:, None]).to(cond_emb.dtype)
            scaler = (cond_emb.shape[1]/denoms)
            pooled_cond_emb *= scaler

        cond_padding_mask[zero_cond_inds] = True
        cond_emb[zero_cond_inds] = 0
        pooled_cond_emb[zero_cond_inds] = 0
        
        # 3. Iterate through layers
        pos_bias = None
        for i, layer in enumerate(self.layers):
            x, pos_bias = layer(x, t_emb, pooled_cond_emb, cond_emb, x_padding_mask, cond_padding_mask, pos_bias=pos_bias)
        # 4. Pass through head to get logits
        x = self.head(x) # (bs, seq_len, vocab size)

        return x
        
if __name__ == '__main__':

    model = TransFusion(ModelConfig(vocab_size=123))
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,d} parameters.")

    x = torch.randint(0, 50, (2, 200))
    t = torch.randint(0, 100, (2,))
    cond_emb = torch.randn(2, 123, model.cfg.cond_emb_dim)
    cond_padding_mask = torch.zeros((2, 123), dtype=torch.bool)

    with torch.no_grad():
        y = model(x, t, cond_emb, cond_padding_mask)
        print(f"{x.shape} --> model --> {y.shape}")
        # char ids of shape (bs, seq_len) --> (bs, seq_len, vocab size) of logits

