import argparse
from dataclasses import dataclass
from pathlib import Path
import logging

import random
from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fastprogress.fastprogress import master_bar, progress_bar
from omegaconf import OmegaConf
from jiwer import wer, cer

from .dataset import PogDataset, collate_batch
from .model import TransFusion
from .diffusion import MultinomialDiffusion, index_to_log_onehot


@dataclass
class DSH():
    # Diffusion Sampling Hyperparameters [DSH] (Section 4)
    jump_len: int = 1 # j in RePaint paper [default 10] (Section 4.1)
    jump_n_sample: int = 1 # r in RePaint paper [default 10] (Section 4.1)
    last_greedy: bool = False # whether to not sample at t=0, but take argmax prediction. [default False]
    x_0_temp: float = 1.0 # reweight temp for model prediction of x0
    guidance_w: float = 1.5 # classifier free guidance weight [default 1.5] (Section 4.3)
    enable_kevin_scaled_inference: bool = True # sequentially progressive diffusion [default True] (Section 4.2)
    T_override: Union[None, int] = 400 # allow variable transcription sizes during inference (Section 4.4)

# ---------------------- REPAINT METHODS -----------------------

def get_schedule(t_T, jump_len=10, jump_n_sample=10):
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t-1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    return ts

def cache(set=False, restore=False, text_targets_all=None, text_preds_all=None, i=None, x_0_preds=None, name=''):
    """ Set cache with `set` or restore cache with `restore`"""
    assert set ^ restore, "either set or restore. not both."

    if set:
        torch.save({
            'text_targets_all': text_targets_all,
            'text_preds_all': text_preds_all,
            'i': i,
            'x_0_preds': x_0_preds
        }, f'tmp_score_cache-{name}.pt')
    elif restore:
        if Path(f'tmp_score_cache-{name}.pt').is_file() == False:
            raise FileNotFoundError("Cache file tmp_score_cache.pt not found in current directory.")
        score_cache = torch.load(f'tmp_score_cache-{name}.pt')
        text_targets_all = score_cache['text_targets_all']
        text_preds_all = score_cache['text_preds_all']
        x_0_preds = score_cache['x_0_preds']
        i = score_cache['i']
    else: raise NotImplementedError()

    return text_targets_all, text_preds_all, i, x_0_preds

# ids to text helper function
def to_text(pp, i2s: list): return ''.join([i2s[p] for p in pp if p != 0])

# text to ids helper function
def to_inds(txt, s2i): return torch.tensor([s2i[ch] for ch in txt], dtype=torch.long)

@torch.inference_mode()
def score_model(model: TransFusion, cfg, dl: DataLoader, vocab, dtype, device, restore, ensemble_size, name=None):
    """ Score a trained `model` on a specific dataset `dl` """

    diff = MultinomialDiffusion(cfg.model_cfg.vocab_size, 
        cfg.model_cfg.T,
        cfg.model_cfg.diffusion_s,
        dtype=dtype,
        device=device
    )

    def forward_diffusion(x, t, c=None):
        log_x_t = index_to_log_onehot(x, cfg.model_cfg.vocab_size, dtype=dtype)
        if c is not None:
            x = diff.q_pred_one_timestep_scaled(log_x_t, t, c, DSH.jump_len)
        else:
            x = diff.q_pred_one_timestep(log_x_t, t)
        x = diff.log_sample_categorical(x)
        return x

    def reverse_diffusion(batch, x_known=None, m=None, last_greedy=False, temperature=1.0, alphas=None, ensemble_size=1):
        """Equation 8: x_{t-1} given x, t, x_known, m. Optionally do not sample model output
        for t=0, but rather use the greedy argmax with `last_greedy`.
        """
        x = batch[0]
        t = batch[1]
        if x_known is None: x_known = torch.zeros_like(x)
        if m is None: m = torch.zeros_like(x)

        # Equation 8b
        x_0_pred = model(*batch)

        if DSH.guidance_w != 1:
            uncond_x_0_pred = model(x, t, torch.zeros_like(batch[2]), torch.ones_like(batch[3]), batch[-1])
            x_0_pred = DSH.guidance_w*x_0_pred + (1-DSH.guidance_w)*uncond_x_0_pred

        x_0_pred = x_0_pred / temperature
        log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)
        log_x_t = index_to_log_onehot(x, diff.num_classes, dtype=x_0_pred.dtype)
        log_model_pred = diff.p_pred(log_x_t, t, log_x_0_pred) # p(x_{t-1} | x_{t})
        
        a_t = alphas[t[0]] if alphas is not None else 0
        mat = torch.eye(ensemble_size, device=x.device)*(1-a_t)
        mat += 1/ensemble_size * a_t
        mat = torch.block_diag(*([mat]*(x.shape[0]//ensemble_size)))
        log_model_pred = ( (mat[..., None, None] ).log().to(x.dtype) + log_model_pred[None])
        log_model_pred = torch.logsumexp(log_model_pred, dim=1)
        
        if (t==0).all() and last_greedy: # Do not sample at t=0
            x_tm1_unknown = log_model_pred.argmax(dim=-1)
        else:
            x_tm1_unknown = diff.log_sample_categorical(log_model_pred)
        
        # Equation 8a
        x_known_log = index_to_log_onehot(x_known, diff.num_classes, dtype=x_0_pred.dtype)
        if (t==0).all(): # Do not sample at t=0
            x_tm1_known = x_known
        else:
            x_tm1_known = diff.q_sample(x_known_log, t)
        
        # Equation 8c
        x_tm1 = x_tm1_known * m.long() + x_tm1_unknown * (1 - m.long())
        return x_tm1, x_0_pred

    if restore:
        text_targets_all, text_preds_all, resume_idx, x_0_preds = cache(restore=True, name=name)
        print(f"Restored eval cache containing {len(text_targets_all)} items computed by index {resume_idx}")
    else:
        text_preds_all = []
        text_targets_all = []
        x_0_preds = None # predictions at t=0 of the model, before log softmax
        resume_idx = 0

    mb = master_bar(enumerate(dl), total=len(dl))

    # ensemble bs (not in paper)
    alphas = torch.linspace(1, 0, cfg.model_cfg.T).to(device)

    for i, batch in mb:
        if i < resume_idx:
            # skip items if we are restoring
            continue

        x, t, cond_emb, x_padding_mask, cond_padding_mask = batch
        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True) # (bs, seq_len)
        cond_emb = cond_emb.to(device, non_blocking=True)
        x_padding_mask = x_padding_mask.to(device, non_blocking=True)
        cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)
        cond_emb = cond_emb.to(dtype)
        batch = (x, t, 
                cond_emb.repeat_interleave(ensemble_size, dim=0), 
                cond_padding_mask.repeat_interleave(ensemble_size, dim=0), 
                x_padding_mask.repeat_interleave(ensemble_size, dim=0))

        # RePaint paper resample scheduling
        times = get_schedule(cfg.model_cfg.T, jump_n_sample=DSH.jump_n_sample, jump_len=DSH.jump_len)

        effective_bs = batch[0].shape[0] * ensemble_size
        x = torch.randint(0, diff.num_classes, (effective_bs, batch[0].shape[-1]), dtype=torch.long, device=batch[0].device)
        x_known = torch.zeros_like(x)
        m = torch.zeros_like(x).bool()

        c = 0 # sequentially progressive diffusion offset (Section 4.2)

        # See RePaint paper algorithm
        for t_last, t_cur in progress_bar(zip(times[:-1], times[1:]), total=len(times)-1, parent=mb):
            if i > resume_idx:
                mb.child.comment = f'CER: {cer(text_targets, text_preds):.3f} | WER: {wer(text_targets, text_preds):.3f}'

            t = torch.ones((effective_bs,), dtype=torch.long, device=x.device) * (t_last)
            if t_cur < t_last:
                if c > DSH.jump_n_sample:
                    c = 0
                c += 1/DSH.jump_len

                # Apply Equation 8 (Main Paper)
                xx = (x, t, batch[2], batch[3], batch[4])
                x, x_0_pred = reverse_diffusion(xx, x_known, m, temperature=DSH.x_0_temp, alphas=alphas, ensemble_size=ensemble_size)
            else:
                # Apply Equation 1 (Main Paper)
                # x = forward_diffusion(x, t)
                if DSH.enable_kevin_scaled_inference:
                    x = forward_diffusion(x, t, c=c)
                else:
                    x = forward_diffusion(x, t, c=None)

        x = x[::ensemble_size]
        x_0_pred = x_0_pred[::ensemble_size]

        text_preds = [to_text(p, vocab['i2s']) for p in x]
        text_targets = [to_text(p, vocab['i2s']) for p in batch[0]]
        text_preds_all.extend(text_preds)
        text_targets_all.extend(text_targets)
        if x_0_preds is None:
            x_0_preds = x_0_pred # (bs, seq_len, vocab_size)
        else:
            x_0_preds = torch.cat((x_0_preds, x_0_pred), dim=0) # (tot+bs, seq_len, vocab_size)

        if i % 25 == 0:
            # print progress
            running_wer = wer(text_targets_all, text_preds_all)
            running_cer = cer(text_targets_all, text_preds_all)
            mb.write(f"[{i:04d}/{len(dl):04d}] running cer: {running_cer:4.3f} | running wer: {running_wer:4.3f}")

            # save cache
            cache(set=True, text_targets_all=text_targets_all, text_preds_all=text_preds_all, i=i, x_0_preds=x_0_preds, name=name)
    
    full_wer = wer(text_targets_all, text_preds_all)
    full_cer = cer(text_targets_all, text_preds_all)
    # Save final cache
    cache(set=True, text_targets_all=text_targets_all, text_preds_all=text_preds_all, i=i, x_0_preds=x_0_preds, name=name)

    return full_cer, full_wer

def main():
    parser = argparse.ArgumentParser(description="Score a trained model on ASR metrics")

    parser.add_argument('--ckpt', required=True, type=str, help="model checkpoint to use.")
    parser.add_argument('--eval_csv', required=True, type=str, help="csv of audio & wavlm paths to eval on.")
    parser.add_argument('--vocab', required=True, type=str, help="path to vocab.pt")
    parser.add_argument('--device', default='cuda', type=str, help="device to use")
    parser.add_argument('--dtype', default='fp16', type=str, choices=['fp16', 'fp32'])
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--ensemble_size', default=1, type=int, help='number of repeated ensembles (not in paper)')
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--uttr_path_contains', default=None, type=str, help='used to limit to dev-clean, dev-other...')
    parser.add_argument('--restore', action='store_true', default=False, help='restore failed eval run from progress saved in tmp_score_cache.pt')

    args = parser.parse_args()
    # load checkpoints
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    csv_pth = Path(args.eval_csv)
    vocab = torch.load(args.vocab)
    
    # load config
    cfg = OmegaConf.structured(ckpt['cfg_yaml'])
    print(f"CKPT CONFIG:\n{OmegaConf.to_yaml(cfg)}")
    print("ARGS: ", args)
    print(f"Diffusion sampling hyperparameters:\n{OmegaConf.to_yaml(OmegaConf.create(DSH))}")

    if not hasattr(cfg.model_cfg, 'attention_type'):
        cfg.model_cfg.attention_type = 'normal'
    if not hasattr(cfg.model_cfg, 'wav_encoder'):
        cfg.model_cfg.wav_encoder = 'wavlm'
        cfg.model_cfg.wavlm_num_bucket = 'wavlm'
        cfg.model_cfg.wavlm_max_dist = 'wavlm'

    # load model
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(device)
    model.load_state_dict(ckpt['module'])
    model.eval()
    print(f"Model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")

    # create dataset
    df = pd.read_csv(csv_pth)
    if args.uttr_path_contains is not None:
        len_pre = len(df)
        df = df[df.audio_path.str.contains(args.uttr_path_contains)] 
        print(f"trimming utterances containing {args.uttr_path_contains}: {len_pre} --> {len(df)}")
    ds = PogDataset(df, vocab['s2i'], vocab['i2s'], cfg.model_cfg.T if DSH.T_override is None else DSH.T_override, cfg.max_transcript_length)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

    # fp16 inference
    if dtype == torch.float16:
        model = model.half()
    
    dl = DataLoader(ds, args.bs, shuffle=False, collate_fn=collate_batch, num_workers=cfg.num_workers)

    word_vocab = None

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #score model
    cer, wer = score_model(model, cfg, dl, vocab, dtype, device, args.restore, args.ensemble_size, args.uttr_path_contains)

    print('-'*50)
    print('\t'*5 + f"{args.ckpt}" + '\t'*5 + '\n')
    print(f"\t CER mean over {args.eval_csv} [{args.uttr_path_contains}]: {cer:4.3f}")
    print(f"\t WER mean over {args.eval_csv} [{args.uttr_path_contains}]: {wer:4.3f}")
    print('-'*50)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main()
