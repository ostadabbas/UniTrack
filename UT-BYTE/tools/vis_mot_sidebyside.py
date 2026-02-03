#!/usr/bin/env python3
"""
Generate side-by-side comparison videos for UT-GTR vs Baseline GTR
FIXED: Properly handles frame mapping for different MOT17 sequences in halfval dataset
"""

import argparse
import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

NC = 1300
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
              for _ in range(NC)]

def get_frame_mapping(seq_name):
    """Get the frame mapping for different MOT17 sequences"""
    frame_mappings = {
        'MOT17-02-FRCNN': 300,  # 600 frames total, validation uses 301-600
        'MOT17-04-FRCNN': 525,  # 1050 frames total, validation uses 526-1050
        'MOT17-05-FRCNN': 418,  # 837 frames total, validation uses 419-837
        'MOT17-09-FRCNN': 262,  # 525 frames total, validation uses 263-525
    }
    return frame_mappings.get(seq_name, 301)  # Default to MOT17-02 if not found

def draw_bbox(img, bboxes, traces, title="", show_trails=False):
    """Draw bounding boxes on image without trails"""
    img_copy = img.copy()
    
    # Add title
    if title:
        cv2.putText(img_copy, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    for bbox in bboxes:
        track_id = int(bbox[0])
        x, y, w, h = bbox[1:5]
        
        # Draw bounding box
        color = COLORS[track_id % len(COLORS)]
        color = tuple(int(c) for c in color)
        
        cv2.rectangle(img_copy, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # Draw track ID
        cv2.putText(img_copy, str(track_id), (int(x), int(y-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_copy

def main():
    parser = argparse.ArgumentParser(description='Generate side-by-side comparison videos')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to ground truth images')
    parser.add_argument('--baseline_preds', type=str, required=True,
                        help='Path to baseline predictions file')
    parser.add_argument('--unitrack_preds', type=str, required=True,
                        help='Path to UniTrack predictions file')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='Sequence name (e.g., MOT17-02-FRCNN)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=10,
                        help='Output video fps')
    
    args = parser.parse_args()
    
    # Get frame mapping for this sequence
    frame_offset = get_frame_mapping(args.seq_name)
    print(f"Using frame offset {frame_offset} for {args.seq_name}")
    
    # Load predictions
    print(f"Loading baseline predictions from {args.baseline_preds}")
    baseline_preds = defaultdict(list)
    with open(args.baseline_preds, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                eval_frame = int(parts[0])  # 1-299 in evaluation
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                baseline_preds[eval_frame].append((track_id, x, y, w, h))
    
    print(f"Loading UniTrack predictions from {args.unitrack_preds}")
    unitrack_preds = defaultdict(list)
    with open(args.unitrack_preds, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                eval_frame = int(parts[0])  # 1-299 in evaluation
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                unitrack_preds[eval_frame].append((track_id, x, y, w, h))
    
    # Get image paths
    img_dir = os.path.join(args.gt_path, args.seq_name, 'img1')
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        sys.exit(1)
    
    # Get available frames
    max_eval_frame = max(max(baseline_preds.keys()) if baseline_preds else [0],
                        max(unitrack_preds.keys()) if unitrack_preds else [0])
    
    print(f"Processing {max_eval_frame} frames")
    
    # Initialize video writer
    # Calculate the first actual frame number
    first_actual_frame = frame_offset + 1
    sample_img_path = os.path.join(img_dir, f'{first_actual_frame:06d}.jpg')
    sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        print(f"Cannot load sample image: {sample_img_path}")
        sys.exit(1)
    
    h, w = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w*2, h))
    
    # Process frames (no trails needed)
    for eval_frame in range(1, max_eval_frame + 1):
        # Convert evaluation frame to actual frame number
        actual_frame = eval_frame + frame_offset
        
        # Load image
        img_path = os.path.join(img_dir, f'{actual_frame:06d}.jpg')
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping frame {eval_frame}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}, skipping frame {eval_frame}")
            continue
        
        # Prepare bounding boxes for this frame
        baseline_bboxes = []
        unitrack_bboxes = []
        
        # Process baseline predictions
        if eval_frame in baseline_preds:
            for track_id, x, y, w, h in baseline_preds[eval_frame]:
                baseline_bboxes.append((track_id, x, y, w, h))
        
        # Process UniTrack predictions
        if eval_frame in unitrack_preds:
            for track_id, x, y, w, h in unitrack_preds[eval_frame]:
                unitrack_bboxes.append((track_id, x, y, w, h))
        
        # Draw predictions without trails
        img_baseline = draw_bbox(img.copy(), baseline_bboxes, {}, 
                               f"Baseline GTR (Frame {actual_frame})", show_trails=False)
        img_unitrack = draw_bbox(img.copy(), unitrack_bboxes, {}, 
                               f"UT-GTR (Frame {actual_frame})", show_trails=False)
        
        # Combine side by side
        combined = np.hstack([img_baseline, img_unitrack])
        
        # Write frame
        out.write(combined)
        
        if eval_frame % 10 == 0:
            print(f"Processed frame {eval_frame}/{max_eval_frame}")
    
    out.release()
    print(f"Video saved to: {args.output}")

if __name__ == "__main__":
    main() 