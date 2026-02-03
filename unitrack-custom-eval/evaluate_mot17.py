#!/usr/bin/env python
"""
Run a CO-MOT checkpoint on MOT-17 (val) and compute the official metrics.
"""
import torch, util.misc as utils
from pathlib import Path
from datasets import build_dataset
from models import build_model
from util.tool import load_model
from engine import evaluate
from main import get_args_parser
import sys
import os

def make_args(ckpt, mot_root, out_dir):
    parser = get_args_parser()
    # ---------- fixed values that mean "MOT-17 evaluation" ----------
    fixed = [
        '--dataset_file', 'e2e_mot',
        '--batch_size',   '1',
        '--g_size',       '3',      # match training config
        '--sampler_lengths', '1',
        '--sample_mode',  'fixed_interval',
        '--sample_interval', '1',
        '--mot_path', mot_root,
        '--output_dir',   out_dir,
        '--resume',       ckpt,
        '--meta_arch',    'motr_unincost',
        '--num_queries',  '60',
        '--query_interaction_layer', 'GQIM',
        '--with_box_refine',
        '--enable_unitrack_loss',
    ]
    return parser.parse_args(fixed)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  required=True, help='checkpoint .pth file')
    p.add_argument('--mot_root', required=True, help='root that contains MOT/MOT17')
    # p.add_argument('--trackeval_path', required=True, help='path to TrackEval/scripts directory')
    p.add_argument('--out', default='outputs/eval_mot17')
    args_cli = p.parse_args()

    # Add TrackEval to sys.path temporarily
    # if args_cli.trackeval_path not in sys.path:
    #     sys.path.insert(0, args_cli.trackeval_path)

    args = make_args(args_cli.ckpt, args_cli.mot_root, args_cli.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- build model and load weights ----------
    model, criterion, postprocessors = build_model(args)
    model = load_model(model, args.resume)
    model.to(device).eval()

    # ---------- build val dataloader ----------
    ds_val   = build_dataset('val', args)
    dl_val   = torch.utils.data.DataLoader(
        ds_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.mot_collate_fn)

    # ---------- run tracker + metrics ----------
    hota = evaluate(model, criterion, postprocessors,
                    dl_val, device, Path(args.output_dir), args)
    print(f'HOTA on MOT-17-val : {hota:.3f}')

if __name__ == '__main__':
    main() 