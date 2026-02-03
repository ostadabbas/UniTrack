"""
https://github.com/xingyizhou/CenterTrack
Modified by Xiaoyu Zhao
Further modified for SportsMOT dataset integration with GTR model

https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py

https://cocodataset.org/#format-data
"""
import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import shutil

# SportsMOT dataset paths
DATA_PATH = "DATASET_ROOT/sportsmot"
OUT_PATH = "DATASET_ROOT/sportsmot/annotations"
os.makedirs(OUT_PATH, exist_ok=True)

# Define splits to process
SPLITS = ["train", "val", "test", "train_half", "val_half"]
HALF_VIDEO = True  # Create half splits for training/validation
CREATE_SPLITTED_ANN = True  # Create split annotation files
USE_DET = False  # Don't use detection files
CREATE_SPLITTED_DET = False  # Don't create split detection files

# Create directory for sequence maps if it doesn't exist
SEQ_MAP_PATH = os.path.join(os.path.dirname(DATA_PATH), 'seq_maps')
os.makedirs(SEQ_MAP_PATH, exist_ok=True)

for split in SPLITS:
    # Skip half splits if they don't exist in the dataset
    if 'half' in split and not HALF_VIDEO:
        continue
        
    # Determine the base split (train, val, test)
    base_split = split.split('_')[0] if 'half' in split else split
    data_path = os.path.join(DATA_PATH, base_split)
    out_path = os.path.join(OUT_PATH, "{}.json".format(split))
    
    # Create output structure for COCO format
    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{
            "id": 1,
            "name": "person"  # Changed from 'player' to 'person' to match codebase expectations
        }]
    }
    # Get video list from directory
    video_list = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    
    # Create a sequence map file for this split
    seq_map_file = os.path.join(SEQ_MAP_PATH, f"sportsmot_{split}.txt")
    with open(seq_map_file, 'w') as f_map:
        for seq in tqdm(sorted(video_list)):
            if ".DS_Store" in seq or not os.path.isdir(os.path.join(data_path, seq)):
                continue
            
            # Write sequence to map file
            f_map.write(f"{seq}\n")
            
            video_cnt += 1  # video sequence number.
            out["videos"].append({"id": video_cnt, "file_name": seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt/gt.txt")
            
            # Check if annotation file exists
            if not os.path.exists(ann_path) and split != "test":
                print(f"Warning: Annotation file not found for {seq}")
                continue
                
            images = os.listdir(img_path)
            num_images = len([image for image in images
                              if "jpg" in image])  # half and half

            if HALF_VIDEO and ("half" in split):
                image_range = [0, num_images // 2] if "train" in split else \
                                [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(
                    os.path.join(data_path,
                                 "{}/img1/{:06d}.jpg".format(seq, i + 1)))
                height, width = img.shape[:2]
                # Use absolute path for file_name to ensure proper tracking
                abs_img_path = os.path.join(data_path, seq, "img1", "{:06d}.jpg".format(i + 1))
                image_info = {
                    "file_name": abs_img_path,  # absolute path to image
                    "id": image_cnt + i + 1,  # image number in the entire training set.
                    "frame_id": i + 1 - image_range[
                        0],  # image number in the video sequence, starting from 1.
                    "prev_image_id": image_cnt +
                    i if i > 0 else -1,  # image number in the entire training set.
                    "next_image_id":
                    image_cnt + i + 2 if i < num_images - 1 else -1,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width
                }
                out["images"].append(image_info)
            
            print("{}: {} images".format(seq, num_images))
            if split != "test":
                det_path = os.path.join(seq_path, "det/det.txt")
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
                if USE_DET:
                    dets = np.loadtxt(det_path, dtype=np.float32, delimiter=",")
                if CREATE_SPLITTED_ANN and ("half" in split):
                    anns_out = np.array([
                        anns[i] for i in range(anns.shape[0])
                        if int(anns[i][0]) - 1 >= image_range[0]
                        and int(anns[i][0]) - 1 <= image_range[1]
                    ], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, "gt/gt_{}.txt".format(split))
                    fout = open(gt_out, "w")
                    for o in anns_out:
                        fout.write(
                            "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".
                            format(int(o[0]), int(o[1]), int(o[2]), int(o[3]),
                                   int(o[4]), int(o[5]), int(o[6]), int(o[7]),
                                   o[8]))
                    fout.close()
                
                if CREATE_SPLITTED_DET and ("half" in split) and USE_DET:
                    dets_out = np.array([
                        dets[i] for i in range(dets.shape[0])
                        if int(dets[i][0]) - 1 >= image_range[0]
                        and int(dets[i][0]) - 1 <= image_range[1]
                    ], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, "det/det_{}.txt".format(split))
                    dout = open(det_out, "w")
                    for o in dets_out:
                        dout.write(
                            "{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n".
                            format(int(o[0]), int(o[1]), float(o[2]), float(o[3]),
                                   float(o[4]), float(o[5]), float(o[6])))
                    dout.close()
                
                print("{} ann images".format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    # SportsMOT dataset uses different category IDs
                    # In SportsMOT, category ID is always 1 for players
                    # Check visibility (8th column) - only keep visible objects
                    if float(anns[i][8]) < 0.25:  # visibility threshold
                        continue
                        
                    # Check if object is to be ignored (6th column)
                    if int(anns[i][6]) != 1:  # whether to ignore
                        continue
                        
                    # In SportsMOT, category ID (7th column) should be 1 for players
                    if int(anns[i][7]) != 1:
                        continue
                        
                    # Set category_id for COCO format
                    category_id = 1  # person (was player)
                    ann = {
                        "id": ann_cnt,
                        "category_id": category_id,
                        "image_id": image_cnt + frame_id,
                        "track_id": track_id,
                        "instance_id": track_id,  # Add instance_id field required by GTR model
                        "bbox": anns[i][2:6].tolist(),
                        "conf": float(anns[i][6]),
                        "iscrowd": 0,
                        "area": float(anns[i][4] * anns[i][5])
                    }
                    out["annotations"].append(ann)
            image_cnt += num_images
    print("loaded {} for {} images and {} samples".format(
        split, len(out["images"]), len(out["annotations"])))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        
    print(f"Created sequence map file: {seq_map_file}")
    
# Create a README file with instructions
readme_path = os.path.join(OUT_PATH, "README.md")
with open(readme_path, "w") as f:
    f.write("# SportsMOT Dataset in COCO Format\n\n")
    f.write("This directory contains SportsMOT dataset converted to COCO format.\n\n")
    f.write("## Files:\n")
    for split in SPLITS:
        if 'half' in split and not HALF_VIDEO:
            continue
        f.write(f"- {split}.json: COCO format annotations for {split} split\n")
    f.write("\n")
    f.write("## Usage with GTR Model:\n")
    f.write("Use these annotations with the GTR_SportsMOT_UniTrack.yaml config file.\n")

print(f"\nConversion complete! COCO format annotations saved to {OUT_PATH}")