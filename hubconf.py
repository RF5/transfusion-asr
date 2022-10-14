dependencies = ['torch', 'torchaudio', 'numpy', 'omegaconf', 'fastprogress', 'pandas', 'jiwer']

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from omegaconf import OmegaConf
from fastprogress.fastprogress import progress_bar
from transfusion.model import TransFusion
from transfusion.diffusion import MultinomialDiffusion, index_to_log_onehot
from transfusion.score import DSH, get_schedule, to_text

from wavlm.WavLM import WavLM, WavLMConfig
from wavlm.extract import WEIGHTINGS


def extract_transfusion_features(wav: Tensor, wavlm: WavLM) -> Tensor:
    """ Convert a 16kHz normalized floating point waveform to TransFusion-compatible WavLM features.
    Concretely, the input:
    - `wav`: (1, T) 16kHz waveform.
    - `wavlm`: WavLM module loaded from wavlm_large()
    Returns:
    - `wavlm_features`: (seq_len, dim)
    """
    weighting = torch.tensor(WEIGHTINGS, device=wav.device)[:, None]

    # extract the representation of each layer
    wav_input_16khz = wav.to(next(wavlm.parameters()).device)
    rep, layer_results = wavlm.extract_features(wav_input_16khz, output_layer=wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
    features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)

    features = ( features*weighting[:, None] ).sum(dim=0) # (seq_len, dim)
    return features


# ---------------------------
# Functions adapted from the full score.py

def forward_diffusion(cfg, diff, dtype, x, t, c=None):
    """Simple forward diffusion process p"""
    log_x_t = index_to_log_onehot(x, cfg.vocab_size, dtype=dtype)
    if c is not None:
        x = diff.q_pred_one_timestep_scaled(log_x_t, t, c, DSH.jump_len)
    else:
        x = diff.q_pred_one_timestep(log_x_t, t)
    x = diff.log_sample_categorical(x)
    return x


def reverse_diffusion(diff, model, batch, x_known=None, m=None, last_greedy=False, temperature=1.0, alphas=None, ensemble_size=1):
    """Reverse diffusion process q: predict x_{t-1} given x, t, x_known, m. Optionally do not sample model output
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


@torch.inference_mode()
def perform_simple_inference(model: TransFusion, cond_emb: Tensor, diff: MultinomialDiffusion, vocab, cfg):
    device = cond_emb.device
    dtype = torch.float32
    bs = cond_emb.shape[0]
    x = torch.randint(0, diff.num_classes, (cond_emb.shape[0], DSH.T_override), dtype=torch.long, device=cond_emb.device)
    cond_emb = cond_emb.to(device, non_blocking=True)
    cond_padding_mask = torch.zeros_like(cond_emb, dtype=torch.bool)[..., 0]
    cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)
    cond_emb = cond_emb.to(dtype)

    # RePaint paper resample scheduling
    times = get_schedule(cfg.T, jump_n_sample=DSH.jump_n_sample, jump_len=DSH.jump_len)

    x_known = torch.zeros_like(x)
    m = torch.zeros_like(x).bool()

    c = 0 # sequentially progressive diffusion offset (Section 4.2)

    # ensemble bs (not in paper)
    alphas = torch.linspace(1, 0, cfg.T).to(device)

    # See RePaint paper algorithm
    for t_last, t_cur in progress_bar(zip(times[:-1], times[1:]), total=len(times)-1):

        t = torch.ones((bs,), dtype=torch.long, device=x.device) * (t_last)
        if t_cur < t_last:
            if c > DSH.jump_n_sample:
                c = 0
            c += 1/DSH.jump_len

            # Reverse diffusion: q
            xx = (x, t, cond_emb, cond_padding_mask, None)
            x, x_0_pred = reverse_diffusion(diff, model, xx, x_known, m, temperature=DSH.x_0_temp, alphas=alphas, ensemble_size=1)
        else:
            # Forward diffusion: p
            if DSH.enable_kevin_scaled_inference:
                x = forward_diffusion(cfg, diff, dtype, x, t, c=c)
            else:
                x = forward_diffusion(cfg, diff, dtype, x, t, c=None)

    text_preds = [to_text(p, vocab['i2s']) for p in x]
    return x, text_preds


# ------------------
# torch hub integration functions

def transfusion_small_462k(pretrained=True, progress=True, device='cuda') -> TransFusion:
    """ Best TransFusion model described in the paper, ~250M parameters and trained for
    462 000 updates. A multinomial diffusion ASR model transcribing utterances from their WavLM embeddings.
    """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    # load checkpoints
    ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion_462k_slim.pt",
        map_location=device,
        progress=progress
    )

    device = torch.device(device)
    vocab = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion-vocab.pt",
        map_location='cpu',
        progress=progress
    )
    
    # load config
    cfg = OmegaConf.structured(ckpt['cfg_yaml'])
    logging.debug(f"CKPT CONFIG:\n{OmegaConf.to_yaml(cfg)}")
    logging.debug(f"Default diffusion sampling hyperparameters:\n{OmegaConf.to_yaml(OmegaConf.create(DSH))}")

    # load model
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(device)
    if pretrained:
        model.load_state_dict(ckpt['module'])
    model.eval()
    print(f"TransFusion-small 462k update model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")

    # create diffusion
    diffuser = MultinomialDiffusion(cfg.model_cfg.vocab_size, 
        cfg.model_cfg.T,
        cfg.model_cfg.diffusion_s,
        device=device
    )

    model.vocab = vocab
    model.diffuser = diffuser
    model.perform_simple_inference = perform_simple_inference
    model.forward_diffusion = forward_diffusion
    model.reverse_diffusion = reverse_diffusion
    return model


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/WavLM-Large.pt",
        map_location=device,
        progress=progress
    )
    
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    model.extract_transfusion_features = extract_transfusion_features
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model
