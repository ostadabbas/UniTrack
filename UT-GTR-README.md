# GTR with UniTrack Loss Integration

This repository contains the Global Tracking Transformers (GTR) model with UniTrack (HINGE) loss integration for improved multi-object tracking.

## Introduction

Global Tracking Transformers (GTR) is a transformer-based tracking system that performs object association within long temporal windows. The UniTrack loss has been integrated to enhance the tracking performance by incorporating:

1. **Tracking score** - Improves ID association accuracy
2. **Spatial consistency** - Enforces consistent object sizes/shapes
3. **Temporal consistency** - Encourages smooth motion trajectories

## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.8 with CUDA
- Detectron2

### Step-by-step Installation

```bash
# Create a conda environment
conda create --name gtr python=3.8 -y
conda activate gtr

# Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# Install Detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

# Clone and setup the repository
git clone https://github.com/yourusername/UT-GTR.git
cd UT-GTR
pip install -r requirements.txt
```

## Dataset Preparation

### MOT17 Dataset Setup

1. **Download and Organize the Dataset**

   Download the MOT17 dataset from [MOTChallenge](https://motchallenge.net/data/MOT17/) and organize it as follows:

   ```
   /path/to/UT-MOTR/datasets/
       mot/
           MOT17/
               train/
                   MOT17-02-FRCNN/
                   ...
               test/
                   MOT17-01-FRCNN/
                   ...
   ```

2. **Create Symbolic Links for Evaluation**

   ```bash
   cd /path/to/UT-MOTR/datasets/mot/MOT17/
   ln -s train trainval
   cd ../../../
   ```

3. **Convert Annotations to COCO Format**

   Run the conversion script to create the train/validation split and convert annotations:

   ```bash
   python tools/convert_mot2coco.py
   ```

   This will create two annotation files:
   - `datasets/mot/MOT17/annotations/train_half_conf0.json`
   - `datasets/mot/MOT17/annotations/val_half_conf0.json`

## Configuration

The UniTrack loss is integrated into the GTR system through the following configuration parameters in `configs/GTR_MOT_FPN.yaml`:

```yaml
MODEL:
  ASSO_HEAD:
    HINGE_ON: True          # Enable UniTrack loss
    HINGE_WEIGHT: 1.0       # Weight for UniTrack loss contribution
    HINGE_IOU_THRESHOLD: 0.5 # IoU threshold for object matching
```

You can adjust these parameters to control how the UniTrack loss affects training:
- `HINGE_ON`: Set to `True` to enable, `False` to disable
- `HINGE_WEIGHT`: Higher values increase the loss contribution
- `HINGE_IOU_THRESHOLD`: Threshold for considering objects as matched

## Running the Demo

To run the demo on a video:

```bash
python demo.py --config-file configs/GTR_MOT_FPN.yaml \
  --video-input path/to/your/video.mp4 \
  --output output/demo_output.mp4 \
  --opts MODEL.WEIGHTS models/MODEL_NAME.pth
```

## Training

### Data Preparation

1. First, prepare the MOT datasets following the [datasets README](datasets/README.md).
2. Download the pre-trained weights for initialization.

### Training on MOT17

To train GTR with UniTrack loss on the MOT17 dataset:

```bash
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --num-gpus 4 \
  DATASETS.TRAIN '("mot_train_half_conf0",)' \
  DATASETS.TEST '("mot_val_half_conf0",)' \
  OUTPUT_DIR path/to/save/model
```

### Using CrowdHuman Pre-training

For better performance, you can first pre-train on CrowdHuman and then fine-tune on MOT17:

```bash
# First, train on CrowdHuman
python train_net.py --config-file configs/GTR_CH_FPN.yaml \
  --num-gpus 4 \
  OUTPUT_DIR path/to/crowdhuman_model

# Then fine-tune on MOT17
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --num-gpus 4 \
  MODEL.WEIGHTS path/to/crowdhuman_model/model_final.pth \
  OUTPUT_DIR path/to/mot17_model
```

To disable UniTrack loss during training, you can modify the config using command-line arguments:

```bash
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --num-gpus 4 \
  MODEL.ASSO_HEAD.HINGE_ON False \
  OUTPUT_DIR path/to/save/model
```

### Adjusting Loss Weights

The balance between association loss and UniTrack loss can be tuned by modifying the weights:

```bash
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --num-gpus 4 \
  MODEL.ASSO_HEAD.ASSO_WEIGHT 1.0 \
  MODEL.ASSO_HEAD.HINGE_WEIGHT 0.5 \
  OUTPUT_DIR path/to/save/model
```

## Evaluation

### MOT17 Evaluation

To evaluate a trained model on the MOT17 validation set:

```bash
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --eval-only \
  DATASETS.TEST '("mot_val_half_conf0",)' \
  MODEL.WEIGHTS path/to/trained/model.pth \
  OUTPUT_DIR path/to/save/results
```

To evaluate on the MOT17 test set (after training on the full training set):

```bash
python train_net.py --config-file configs/GTR_MOT_FPN.yaml \
  --eval-only \
  DATASETS.TEST '("mot_test",)' \
  MODEL.WEIGHTS path/to/trained/model.pth \
  OUTPUT_DIR path/to/save/results
```

### MOT17 Metrics

The evaluation will report standard MOT metrics including:
- MOTA: Multi-Object Tracking Accuracy
- IDF1: ID F1 Score (measures track identity consistency)
- MT: Mostly Tracked targets
- ML: Mostly Lost targets
- FP: False Positives
- FN: False Negatives
- IDS: ID Switches

The UniTrack loss particularly aims to improve the IDF1 and IDS metrics.

### TAO Dataset Evaluation

For TAO dataset evaluation:

```bash
python train_net.py --config-file configs/GTR_TAO_DR2101.yaml \
  --eval-only \
  MODEL.WEIGHTS path/to/trained/model.pth \
  OUTPUT_DIR path/to/save/results
```

## Monitoring UniTrack Loss

During training, the UniTrack loss components are logged separately:
- `loss_hinge`: Overall UniTrack loss
- `loss_hinge_tracking`: Tracking score component
- `loss_hinge_spatial`: Spatial consistency component
- `loss_hinge_temporal`: Temporal consistency component

You can monitor these values in the training logs to understand how each component contributes to the overall performance.

## Visualizing Results

After running the demo, you can visualize the tracking results:

```bash
python demo.py --config-file configs/GTR_MOT_FPN.yaml \
  --video-input path/to/your/video.mp4 \
  --output output/demo_output.mp4 \
  --opts MODEL.WEIGHTS path/to/trained/model.pth
```

## References

- [Global Tracking Transformers](http://arxiv.org/abs/2203.13250) (CVPR 2022)
- UniTrack (HINGE) loss for improved tracking consistency and accuracy
