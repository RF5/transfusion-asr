import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress.fastprogress import master_bar
from matplotlib import colors
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from transfusion.diffusion import MultinomialDiffusion, index_to_log_onehot


# Algorithm 3 (including returning all images)
@torch.inference_mode()
def eval_multinomial_cer(
    steps: int,
    valid_idx: int,
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    model: nn.Module,
    diff: MultinomialDiffusion, 
    T: int = 200,
    plot: bool = False,
    logger: SummaryWriter = None,
    i2s: dict = None,
    mb = None
):
    """ Evaluate a single batch given a model
    - `steps`: global training step number
    - `valid_idx`: index of batch in validation set
    - `batch`: collated batch of (x, t, cond_emb, cond_padding_mask, x_padding_mask)
        - `x`: (bs, seq_len) long tensor of character indices
        - `t`: (bs, ) long tensor of timestep indices
        - `cond_emb`: (bs, seq_len2, cond_emb_dim)
        - `cond_padding_mask`: (bs, seq_len2)
        - `x_padding_mask`: (bs, seq_len)

    - `model`: model to evaluate batch
    - `T`: int max decoding steps
    - `vocab_size`: int size of vocabulary
    - `log`: bool log to tensorbard
    - `logger`: Tensorboard SummaryWriter

    Returns mean cer per batch item
    """
    y = batch[0]
    
    # Filter CER by ground truth transcript length
    idx = y.nonzero()

    b = y.shape[0]
    # start from pure noise (for each example in the batch)
    x = torch.randint(0, diff.num_classes, y.shape, dtype=torch.long, device=y.device)
    
    cer = []
    # use first item in batch
    transitions = [x[0].cpu().numpy()]

    for i in reversed(range(0, T)):
        t = torch.ones((b,), dtype=torch.long, device=x.device) * (i)
        xx = (x, t, batch[2], batch[3], batch[4])
 
        x_0_pred = model(*xx)
        log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)

        log_x_t = index_to_log_onehot(x, diff.num_classes, dtype=x_0_pred.dtype)
        log_model_pred = diff.p_pred(log_x_t, t, log_x_0_pred) # p(x_{t-1} | x_{t})
        x = diff.log_sample_categorical(log_model_pred)
        preds = x

        if mb is not None: mb.child.comment = f"step {i:03d}/{T:03d}"

        if plot:
            cer.append(1-((y[0, idx[idx[:,0]==0,1]]==preds[0, idx[idx[:,0]==0,1]]).float().mean()).cpu().numpy())
            transitions.append(preds[0].cpu().numpy())
    
    if plot:
        fig = _draw_cer(T, cer)
        logger.add_figure(f'eval/denoise_cer/{valid_idx}', fig, steps)
        logger.flush()
        
        fig = _draw_denoise(y[0].cpu().numpy(), transitions[::-1], i2s=i2s)
        logger.add_figure(f'eval/denoising_sample/{valid_idx}', fig, steps)

    c = (y==preds).float()
    idx = (y==0).nonzero()
    c[idx[:,0],idx[:,1]] = 0
    fixed_cer = 1 - c.sum(dim=-1)/ ( (y != 0).sum(dim=-1) )

    return 1-(y==preds).float().mean(dim=-1), fixed_cer


def _draw_cer(
    T: int,
    cer: list[float],
):
    fig = plt.Figure()
    ax = fig.gca()
    ax.plot(np.arange(T), cer)
    ax.set_title('CER to denoising steps')
    ax.set_xlabel('denoising step')
    ax.set_ylabel('CER')
    return fig

def _draw_denoise(
    label: np.ndarray,
    transitions: list[np.ndarray],
    i2s: dict = None,
    eps: str = 'eps',
):
    dim = (transitions[0].shape[0], len(transitions))

    fig = plt.figure(figsize=((dim[1]+2)/4, (dim[0]+3)/4))

    pad = 3
    ax = []
    ax.append(plt.axes([pad/(dim[1]+pad*2), 0, dim[1]/(dim[1]+pad*2), 1]))
    ax.append(plt.axes([(pad+dim[1]+1)/(dim[1]+pad*2), 0, 1/(dim[1]+pad*2), 1]))

    bg = np.zeros(dim)

    bg[:,0] = (transitions[-1] == label)+2
    cmap = colors.ListedColormap(['white', 'red', 'yellow', 'green'])
    bounds = np.arange(-1,4,1)+0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)
    for z in range(dim[0]):
        for x in range(dim[1]):
            c = transitions[dim[1]-x-1][z]
            ch = i2s[c]
            if ch == eps: ch = '\u03B5'
            ax[0].text((x+0.3), (dim[0]-z-0.7)/dim[0], ch, family='monospace')
            if x>0 and c != transitions[dim[1]-x][z] and c == label[z]:
                bg[z,x] = 3
            elif x>0 and c != transitions[dim[1]-x][z] and transitions[dim[1]-x][z] == label[z]:
                bg[z,x] = 1
            elif x>0 and c != transitions[dim[1]-x][z] and c != label[z]:
                bg[z,x] = 2

    x=0
    for z in range(dim[0]):
        ch = i2s[label[z]]
        if ch == eps: ch = '\u03B5'
        ax[1].text((x+0.3), (dim[0]-z-0.7)/dim[0], ch, family='monospace')

    ax[0].set_xlim((0,dim[1]))
    ax[0].set_xlabel('diffusion step')
    ax[0].set_ylabel('text transcript')
    ax[0].imshow(bg, extent=[0,dim[1],0,1], cmap=cmap, aspect=dim[0], norm=norm)

    ax[1].imshow(np.ones((dim[1],1))*3, extent=[0,1,0,1], cmap=cmap, norm=norm, aspect=dim[0])
    ax[1].set_xlim((0,1))
    ax[1].set_xlabel('label')
    ax[1].set_yticks([])

    ax[0].set_title('Denoising process')

    return fig
