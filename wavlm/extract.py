
""" Utility script to extract WavLM embeddings. """

import pandas as pd
from pathlib import Path
from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import argparse
import torch
import torchaudio
from .WavLM import WavLM, WavLMConfig

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
WEIGHTINGS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, # layer 15
    1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9 # layer 16-24
]

def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    for f in folders:
        all_files.extend(list((root_path/f).rglob('**/*.flac')))
    speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def get_wavlm(path, device) -> WavLM:
    checkpoint = torch.load(str(path), map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model

@torch.inference_mode()
def extract(df: pd.DataFrame, model: WavLM, device, ls_path: Path, out_path: Path, meanpool: bool):
    pb = progress_bar(df.iterrows(), total=len(df))
    weighting = torch.tensor(WEIGHTINGS, device=device)[:, None]

    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (out_path/rel_path).with_suffix('.pt')
        if targ_path.is_file(): continue
        os.makedirs(targ_path.parent, exist_ok=True)

        x, sr = torchaudio.load(row.path)

        # extract the representation of each layer
        wav_input_16khz = x.to(device)
        rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)

        if meanpool:
            # 1. take mean over sequence length
            features = features.mean(dim=1)
            # 2. take weighted mean over layers
            features = ( features*weighting ).sum(dim=0) # (dim,)
        else:
            # save full sequence
            features = ( features*weighting[:, None] ).sum(dim=0) # (seq_len, dim)
        if i == 0: print("Feature has shape: ", features.shape, flush=True)
        # 3. save
        torch.save(features.cpu(), str(targ_path))
        pb.comment = str(rel_path)
        pb.wait_for = min(pb.wait_for, 10)

        if i % 1000 == 0: print(f"Done {i:,d}/{len(df):,d}", flush=True)
        

def main():
    parser = argparse.ArgumentParser(description="Compute WavLM features for the librispeech dataset")

    parser.add_argument('--librispeech_path', required=True, type=str, help="root path of librispeech dataset")
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--out_path', required=True, type=str, help="target directory to save WavLM features into")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ckpt_path', default=f'{THIS_DIR}/WavLM-Large.pt', type=str, help='Path to pretrained WavLM checkpoint')
    parser.add_argument('--meanpool', action='store_true', default=False, help="Only save mean WavLM feature over sequence length for each utterance.")

    args = parser.parse_args()
    print(f"Weightings: {WEIGHTINGS}\nWeightings sum: {sum(WEIGHTINGS)}")
    print("Mean pool: ", args.meanpool)

    if args.librispeech_path is not None:
        ls_df = make_librispeech_df(Path(args.librispeech_path))

    model = get_wavlm(args.ckpt_path, args.device)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extract(ls_df, model, args.device, Path(args.librispeech_path), Path(args.out_path), args.meanpool)

if __name__ == '__main__':
    main()