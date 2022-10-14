
import pandas as pd
from pathlib import Path
from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import argparse
import torch

THIS_DIR = Path(__file__).parent


def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    for f in folders:
        all_files.extend(list((root_path/f).rglob('**/*.flac')))
    speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
    subset = [f.parents[2].stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers, 'subset': subset})
    return df

def get_transcriptions(df: pd.DataFrame) -> pd.DataFrame:
    transcripts = {}
    out_transcripts = []
    for i, row in progress_bar(df.iterrows(), total=len(df)):
        p = Path(row.path)
        if p.stem in transcripts:
            out_transcripts.append(transcripts[p.stem])
        else:

            with open(p.parent/f'{p.parents[1].stem}-{p.parents[0].stem}.trans.txt', 'r') as file:
                lines = file.readlines()
                for l in lines:
                    uttr_id, transcrip = l.split(' ', maxsplit=1)
                    transcripts[uttr_id] = transcrip.strip()
            out_transcripts.append(transcripts[p.stem])
    df['transcription'] = out_transcripts
    return df

def get_wavlm_feat_paths(df: pd.DataFrame, ls_path, wavlm_path) -> pd.DataFrame:
    pb = progress_bar(df.iterrows(), total=len(df))
    targ_paths = []
    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (wavlm_path/rel_path).with_suffix('.pt')
        assert targ_path.is_file()
        targ_paths.append(targ_path)
    df['wavlm_path'] = targ_paths
    return df

def get_vocab(df: pd.DataFrame, eps_idx=0):
    vocab = set(('eps',))
    for i, row in progress_bar(df.iterrows(), total=len(df)):
        chars = list(str(row.transcription).strip().upper())
        vocab |= set(chars)
    vocab = sorted(list(vocab), key=lambda x: ord(x) if x != 'eps' else -1)
    return vocab

def main():
    parser = argparse.ArgumentParser(description="Generate train & valid csvs from dataset directories")

    parser.add_argument('--librispeech_path', required=True, type=str, help="path to root of librispeech dataset")
    parser.add_argument('--ls_wavlm_path', required=True, type=str, help="path to root of WavLM features extracted using extract.py")
    parser.add_argument('--include_test', action='store_true', default=False, help="include processing and saving test.csv for test subsets")

    args = parser.parse_args()

    if args.librispeech_path is not None:
        ls_df = make_librispeech_df(Path(args.librispeech_path))

    ls_df = get_transcriptions(ls_df)

    ls_df = get_wavlm_feat_paths(ls_df, Path(args.librispeech_path), Path(args.ls_wavlm_path))
    ls_df.rename(columns={'path': 'audio_path'}, inplace=True)
    train_csv = ls_df[ls_df.subset.str.contains('train')]
    valid_csv = ls_df[ls_df.subset.str.contains('dev')]
    train_csv = train_csv.sort_values('audio_path')
    valid_csv = valid_csv.sort_values('audio_path')
    
    os.makedirs('splits/', exist_ok=True)
    train_csv.to_csv('splits/train.csv', index=False)
    valid_csv.to_csv('splits/valid.csv', index=False)
    print(f"Saved train csv (N={len(train_csv)}) and valid csv (N={len(valid_csv)} to splits/")

    if args.include_test:
        test_csv = ls_df[ls_df.subset.str.contains('test')]
        test_csv = test_csv.sort_values('audio_path')
        test_csv.to_csv('splits/test.csv', index=False)
        print(f"Saved test csv (N={len(test_csv)}) to splits/test.csv")

    # save vocab as well, fairseq style
    vocab = get_vocab(ls_df)
    i2s = vocab
    s2i = {s: i for i, s in enumerate(vocab)}
    torch.save({'i2s': i2s, 's2i': s2i}, 'splits/vocab.pt')
    print("Vocab: ", s2i)

if __name__ == '__main__':
    main()