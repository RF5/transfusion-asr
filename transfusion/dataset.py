import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
import random
import pandas as pd
import torchaudio


class PogDataset(Dataset):

    def __init__(self, df: pd.DataFrame, s2i: dict[str: int], i2s: list | Tensor, T: int, 
                 max_transcript_length: int, use_text_bypass: bool = False) -> None:
        """ `use_text_bypass` determines whether the WavLM inputs are replaced by the ground
        truth text inputs for sanity checking purposes.
        """
        super().__init__()
        self.df = df
        self.s2i = lambda x: s2i[x]
        self.i2s = lambda x: i2s[x]
        self.T = T
        self.max_transcript_len = max_transcript_length
        self.use_text_bypass = use_text_bypass
        if use_text_bypass:
            print("USING TEXT BYPASS FOR SANITY CHECKING")
    
    def __len__(self): return len(self.df)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.df.iloc[index]

        text = [self.s2i(x) for x in list(row.transcription.strip().upper())]

        if self.use_text_bypass:
            wavlm_feat = torch.tensor(text, dtype=torch.long)
            wavlm_feat = F.one_hot(wavlm_feat, num_classes=len(self.i2s))
        else: wavlm_feat = torch.load(row.wavlm_path, map_location='cpu') # (seq_len, dim)

        n_pad = self.max_transcript_len - len(text)
        if n_pad > 0:
            text += [self.s2i('eps'),]*n_pad
        elif n_pad < 0:
            text = text[:self.max_transcript_len]
        text = torch.tensor(text, dtype=torch.long)

        t = torch.tensor(random.randint(0, self.T-1), dtype=torch.long)  # sample t \in [0, T-1], aka [0, T)

        return text, t, wavlm_feat
        

def collate_batch(
    batch: list[tuple[Tensor, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ Batch collate function, pads WavLM embeddings and creates attention mask
    - `batch`: list of (transcript, t, WavLM feature sequence)
        - transcript: (seq_len) long tensor
        - t: (1,) long tensor
        - WavLM: (seq_len2, dim) fp32 tensor, arbitrary seq_len2; or (T)

    Returns (x, t, cond_emb, x_padding_mask, cond_padding_mask)
    - `x`: (bs, seq_len) long tensor of character indices
    - `t`: (bs, ) long tensor of timestep indices
    - `cond_emb`: (bs, seq_len2, cond_emb_dim) or (bs, T)
    - `x_padding_mask`: (bs, seq_len)
    - `cond_padding_mask`: (bs, seq_len2) or (bs, T)
    """

    x, t, cond_emb =  zip(*batch, strict=True)

    x = torch.stack(x)
    t = torch.stack(t)

    x_padding_mask = torch.zeros_like(x, dtype=torch.bool)

    ll = torch.tensor([x.shape[0] for x in cond_emb], dtype=torch.long)

    cond_emb = torch.nn.utils.rnn.pad_sequence(cond_emb, batch_first=True) 
    cond_padding_mask = torch.arange(cond_emb.shape[1], dtype=torch.long)[None, :] >= ll[:, None]

    return x, t, cond_emb, x_padding_mask, cond_padding_mask



