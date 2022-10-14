from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def flatten_cfg(cfg: DictConfig | ListConfig) -> dict:
    """ 
    Recursively flattens a config into a flat dictionary compatible with 
    tensorboard's `add_hparams` function.
    """
    out_dict = {}
    if type(cfg) == ListConfig:
        cfg = DictConfig({f"[{i}]": v for i, v in enumerate(cfg)})

    for key in cfg:
        if type(getattr(cfg, key)) in (int, str, bool, float):
            out_dict[key] = getattr(cfg, key)
        elif type(getattr(cfg, key)) in [DictConfig, ListConfig]:
            out_dict = out_dict | {f"{key}{'.' if type(getattr(cfg, key)) == DictConfig else ''}{k}": v for k, v in flatten_cfg(getattr(cfg, key)).items()}
        else: raise AssertionError
    return out_dict

def lin_one_cycle(startlr, maxlr, endlr, warmup_pct, total_iters, iters):
    """ 
    Linearly warms up from `startlr` to `maxlr` for `warmup_pct` fraction of `total_iters`, 
    and then linearly anneals down to `endlr` until the final iter.
    """
    warmup_iters = int(warmup_pct*total_iters)
    if iters < warmup_iters:
        # Warmup part
        m = (maxlr - startlr)/warmup_iters
        return m*iters + startlr
    else:
        m = (endlr - maxlr)/(total_iters - warmup_iters)
        c = endlr - total_iters*m
        return m*iters + c   