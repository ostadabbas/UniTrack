#!/usr/bin/env python
"""
Run a CO-MOT checkpoint on MOT-17 (train) and generate tracking results.
This version properly combines multiple tracker groups (g_size=3) into a single result.
No metric computation is performed.
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
    # ---------- fixed values that mean "MOT-17 train data generation" ----------
    fixed = [
        '--dataset_file', 'e2e_mot',
        '--meta_arch',    'motr_unincost',
        '--batch_size',   '1',
        '--g_size',       '3',      # match training config for model loading
        '--sampler_lengths', '1',
        '--sample_mode',  'random_interval',
        '--sample_interval', '1',
        '--mot_path',     mot_root,
        '--resume',       ckpt,
        '--output_dir',   out_dir,
        '--with_box_refine',
        '--query_interaction_layer', 'GQIM',
        '--num_queries',  '60',
        '--enable_unitrack_loss',
    ]
    
    return parser.parse_args(fixed)

def combine_tracker_groups(dt_instances_all, g_size=3, combination_method='best_score'):
    """
    Combine multiple tracker groups into a single output.
    
    Args:
        dt_instances_all: All detections from all groups
        g_size: Number of groups  
        combination_method: How to combine groups ('best_score', 'nms', 'voting')
    """
    import torch
    if combination_method == 'best_score':
        # For each unique object ID, keep the detection with highest score across all groups
        active_indx = []
        full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
        
        for id in torch.unique(dt_instances_all.obj_idxes):
            if id < 0:  # Skip invalid IDs
                continue
            indx = torch.where(dt_instances_all.obj_idxes == id)[0]
            best_idx = full_indx[indx][dt_instances_all.scores[indx].argmax()]
            active_indx.append(best_idx)
        
        if len(active_indx):
            active_indx = torch.stack(active_indx)
            return dt_instances_all[active_indx]
        else:
            return dt_instances_all[:0]  # Empty result
    
    elif combination_method == 'group_separate':
        # Use only group 0 (first group) - simplest approach
        return dt_instances_all[dt_instances_all.group_ids == 0]
    
    else:
        raise NotImplementedError(f"Combination method {combination_method} not implemented")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  required=True, help='checkpoint .pth file')
    p.add_argument('--mot_root', required=True, help='root that contains MOT/MOT17')
    p.add_argument('--out', default='outputs/eval_mot17_train_combined')
    args_cli = p.parse_args()

    print("Building model with g_size=3 to match training configuration...")
    args = make_args(args_cli.ckpt, args_cli.mot_root, args_cli.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- build model and load weights ----------
    model, criterion, postprocessors = build_model(args)
    model = load_model(model, args.resume)
    model.to(device).eval()

    # ---------- build train dataloader (modified to use train data) ----------
    # Temporarily modify args to use train data
    args.use_train_data = True
    ds_train = build_dataset_train(args)
    
    # Create a simple collate function that doesn't interfere with our data
    def simple_collate_fn(batch):
        return batch[0]  # Just return the single item since batch_size=1
    
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=1, shuffle=False, num_workers=0,  # Set num_workers=0 to avoid multiprocessing issues
        collate_fn=simple_collate_fn)

    print(f"\nNote: Model trained with g_size=3, combining groups for unified output")
    print("Combination method: best_score (highest confidence per object ID)")
    print("Generating tracking results for MOT17 train sequences...")
    
    # ---------- run tracker (no metrics computation) ----------
    generate_tracking_results(model, criterion, postprocessors,
                             dl_train, device, Path(args.output_dir), args)
    print('Tracking results generation completed!')

def build_dataset_train(args):
    """
    Build dataset for MOT17 train and test data by modifying the dataset loading
    """
    import os
    from pathlib import Path
    from collections import defaultdict
    
    # Create a simple dataset class for train data inference
    class DetMOTTrainForInference:
        def __init__(self, args):
            # Override the data path to point to train sequences
            mot_path = args.mot_path
            self.args = args
            self.mot_path = mot_path
            self.labels_full = defaultdict(dict)
            self.video_dict = {}
            self.vid_tmax = {}
            
            # Process both train and test sequences
            data_splits = [
                ('DATASET_ROOT/MOT/MOT17/seqmaps/mot17-train-all.txt', 'train'),
                ('DATASET_ROOT/MOT/MOT17/seqmaps/MOT17-test.txt', 'test')  # Assuming this exists
            ]
            
            # If test seqmap doesn't exist, try to find test sequences directly
            if not os.path.exists('DATASET_ROOT/MOT/MOT17/seqmaps/MOT17-test.txt'):
                print("Test seqmap file not found, searching test directory directly...")
                test_dir = os.path.join(mot_path, 'MOT/MOT17/test')
                if os.path.exists(test_dir):
                    test_sequences = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                    data_splits = [
                        ('DATASET_ROOT/MOT/MOT17/seqmaps/mot17-train-all.txt', 'train'),
                        (test_sequences, 'test')
                    ]
            
            for seqmap_or_list, split_name in data_splits:
                if isinstance(seqmap_or_list, list):
                    # Direct list of sequences
                    sequences = seqmap_or_list
                    print(f"Processing {len(sequences)} {split_name} sequences from directory...")
                else:
                    # Read from seqmap file
                    if not os.path.exists(seqmap_or_list):
                        print(f"Warning: {seqmap_or_list} does not exist, skipping {split_name}")
                        continue
                    
                    sequences = []
                    with open(seqmap_or_list, 'r') as f:
                        lines = f.readlines()[1:]  # Skip header
                        sequences = [line.strip() for line in lines if line.strip()]
                    print(f"Processing {len(sequences)} {split_name} sequences from seqmap...")
                
                # Process each sequence
                for seq_name in sequences:
                    # Process all sequences - don't skip DPM/FRCNN anymore
                    vid_path = f'MOT/MOT17/{split_name}/{seq_name}'
                    full_path = os.path.join(mot_path, vid_path)
                    
                    if not os.path.exists(full_path):
                        print(f'Warning: {full_path} does not exist, skipping...')
                        continue
                    
                    # Get frame range by listing images
                    img_dir = os.path.join(full_path, 'img1')
                    if os.path.exists(img_dir):
                        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
                        if img_files:
                            frame_nums = sorted([int(os.path.splitext(f)[0]) for f in img_files])
                            self.vid_tmax[vid_path] = max(frame_nums)
                            # Initialize empty labels (we don't need GT for inference)
                            for frame_num in frame_nums:
                                self.labels_full[vid_path][frame_num] = []
                            print(f'Added sequence: {seq_name} ({split_name}) with {len(frame_nums)} frames')
        
        def __getitem__(self, idx):
            vid = list(self.vid_tmax.keys())[idx]
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            full_video_path = os.path.join(self.mot_path, vid)
            return {
                'video_name': full_video_path,
                'video_min': t_min,
                'video_max': t_max,
            }

        def __len__(self):
            return len(self.vid_tmax)
    
    return DetMOTTrainForInference(args)

def generate_tracking_results(model, criterion, postprocessors, data_loader, device, output_dir, args):
    """
    Generate tracking results without computing metrics.
    This is a simplified version of evaluate_combined without the evaluation part.
    """
    from engine import ListImgDataset, filter_dt_by_score, filter_dt_by_area
    from torch.utils.data import DataLoader
    from collections import defaultdict
    from copy import deepcopy
    import os
    import torch
    
    model.eval()
    criterion.eval()
    predict_path = os.path.join(output_dir, 'tracker')  # Single tracker output
    prob_threshold = 0.5
    area_threshold = 100

    total_sequences = len(data_loader)
    for seq_idx, data_dict in enumerate(data_loader):
        # Since we're using our simple collate function, data_dict is already the dict, not a batch
        video_name = data_dict['video_name']
        seq_num = os.path.basename(video_name)
        img_dir = os.path.join(video_name, 'img1')
        
        print(f"Processing sequence {seq_idx+1}/{total_sequences}: {seq_num}")
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist, skipping sequence {seq_num}")
            continue
            
        img_list = os.listdir(img_dir)
        img_list = [os.path.join(img_dir, i) for i in img_list if 'jpg' in i]
        img_list = sorted(img_list)
        
        # Clear model state
        try:
            model.module.srcs = []
        except:
            try:
                model.srcs = []
            except:
                print('model have not srcs!')
        
        track_instances = None
        det_db = []
        loader = DataLoader(ListImgDataset('', img_list, det_db), 1, num_workers=2)
        lines = []  # Single list instead of defaultdict for groups
        total_dts = 0
        
        for i, data in enumerate(loader):
            if i % 100 == 0:  # Print progress every 100 frames
                print(f"  Processing frame {i+1}/{len(loader)}")
                
            cur_img, ori_img, proposals, f_path = [d[0] for d in data]
            cur_img, proposals = cur_img.to(device), proposals.to(device)

            if track_instances is not None:
                track_instances.remove('boxes')
            seq_h, seq_w, _ = ori_img.shape

            # Run inference
            try: 
                res = model.module.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            except:
                res = model.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances)

            # Filter by score and area (same as original)
            dt_instances_all = filter_dt_by_score(dt_instances_all, prob_threshold)
            dt_instances_all = filter_dt_by_area(dt_instances_all, area_threshold)
            
            # Apply the same deduplication logic as original (lines 322-329)
            active_indx = []
            full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
            for id in torch.unique(dt_instances_all.obj_idxes):
                indx = torch.where(dt_instances_all.obj_idxes == id)[0]
                active_indx.append(full_indx[indx][dt_instances_all.scores[indx].argmax()])
            if len(active_indx):
                active_indx = torch.stack(active_indx)
                dt_instances_all = dt_instances_all[active_indx]
            
            # **KEY CHANGE**: Instead of saving for each group, combine and save once
            dt_instances = combine_tracker_groups(dt_instances_all, g_size=args.g_size, combination_method='best_score')
            
            total_dts += len(dt_instances)

            # Save format (same as original)
            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()
            
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                frame_ith = int(os.path.splitext(os.path.basename(f_path))[0])
                lines.append(save_format.format(frame=frame_ith, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    
        # Save single combined output (instead of multiple group outputs)
        os.makedirs(predict_path, exist_ok=True)
        with open(os.path.join(predict_path, f'{seq_num}.txt'), 'w') as f:
            f.writelines(lines)
        print("{}: totally {} combined detections".format(seq_num, total_dts))

    print(f"Tracking results saved to: {predict_path}")

if __name__ == '__main__':
    main() 