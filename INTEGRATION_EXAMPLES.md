# UniTrack Integration Examples

This document provides detailed integration examples for all 7 MOT frameworks where UniTrack has been successfully integrated. Each example shows the specific code changes needed and configuration settings used.

## Table of Contents

1. [GTR (Global Tracking Transformers)](#1-gtr-global-tracking-transformers)
2. [MOTR (MOT with Transformers)](#2-motr-mot-with-transformers)
3. [TrackFormer](#3-trackformer)
4. [ByteTrack](#4-bytetrack)
5. [FairMOT](#5-fairmot)
6. [MOTE](#6-mote)
7. [MOTR-SIG](#7-motr-sig)

---

## 1. GTR (Global Tracking Transformers)

**Framework Type**: Transformer-based with ROI pooling  
**Integration Difficulty**: Medium  
**Location**: `UT-GTR/`

### Integration Steps

#### Step 1: Add the Loss Module

Copy `unitrack_criterion.py` to `UT-GTR/gtr/modeling/`:

```bash
cp unitrack_criterion.py UT-GTR/gtr/modeling/
```

#### Step 2: Modify ROI Heads

Edit `UT-GTR/gtr/modeling/roi_heads/gtr_roi_heads.py`:

```python
# Add import at the top
from ..unitrack_criterion import Unitrackrion

class GTRROIHeads(CascadeROIHeads):
    def _init_asso_head(self, cfg, input_shape):
        # ... existing initialization code ...
        
        # Add UniTrack initialization
        self.unitrack_on = cfg.MODEL.ASSO_HEAD.get('unitrack_ON', False)
        self.unitrack_weight = cfg.MODEL.ASSO_HEAD.get('unitrack_WEIGHT', 0.0)
        if self.unitrack_on and self.unitrack_weight > 0:
            self.unitrack_criterion = Unitrackrion(
                img_size=cfg.INPUT.get('SIZE', (1920, 1080)),
                iou_threshold=cfg.MODEL.ASSO_HEAD.get('unitrack_IOU_THRESHOLD', 0.5)
            )
            print(f"Initialized UniTrack loss with weight {self.unitrack_weight}")
    
    def _forward_asso(self, features, instances, targets=None):
        # ... existing forward code ...
        
        if self.training:
            # ... existing loss computation ...
            losses = {'loss_asso': self.asso_weight * asso_loss}
            
            # Add UniTrack loss
            if hasattr(self, 'unitrack_on') and self.unitrack_on and self.unitrack_weight > 0:
                # Extract track IDs from association matrices
                pred_track_ids = self._extract_track_ids(asso_outputs[-1], n_t)
                
                # Prepare outputs for UniTrack
                unitrack_outputs = {
                    'pred_boxes': pred_box.view(1, -1, 4),
                    'pred_logits': torch.cat([p.objectness_logits for p in proposals], dim=0).view(1, -1, 1),
                    'track_ids': pred_track_ids.view(1, -1)
                }
                
                # Format targets
                unitrack_targets = [{
                    'boxes': target_box,
                    'track_ids': target_inst_id
                }]
                
                # Compute UniTrack loss
                unitrack_losses = self.unitrack_criterion(unitrack_outputs, unitrack_targets)
                
                # Add to total losses
                for k, v in unitrack_losses.items():
                    losses[k] = self.unitrack_weight * v
            
            return losses
```

#### Step 3: Configuration

Create or modify `configs/GTR_MOT_UniTrack.yaml`:

```yaml
_BASE_: "Base-CenterNet.yaml"
MODEL:
  META_ARCHITECTURE: "GTRRCNN"
  WEIGHTS: 'models/CH_FPN_1x.pth'
  ASSO_ON: True
  ASSO_HEAD:
    ASSO_THRESH: 0.3
    ASSO_THRESH_TEST: 0.55
    ASSO_WEIGHT: 1.0
    # UniTrack configuration
    unitrack_ON: True
    unitrack_WEIGHT: 1.5
    unitrack_IOU_THRESHOLD: 0.4

SOLVER:
  MAX_ITER: 8000
  BASE_LR: 0.00005
  IMS_PER_BATCH: 4

INPUT:
  TRAIN_SIZE: 1280
  TEST_SIZE: 1560
  VIDEO:
    TRAIN_LEN: 8
    TEST_LEN: 32

DATASETS:
  TRAIN: ("mot_train_half_conf0",)
  TEST: ("mot_val_half_conf0",)
```

#### Step 4: Training

```bash
cd UT-GTR
python train_net.py --config-file configs/GTR_MOT_UniTrack.yaml \
  --num-gpus 4 \
  OUTPUT_DIR output/gtr_mot_unitrack
```

---

## 2. MOTR (MOT with Transformers)

**Framework Type**: End-to-end transformer  
**Integration Difficulty**: Easy  
**Location**: `UT-MOTR/`

### Integration Steps

#### Step 1: Add the Loss Module

```bash
cp unitrack_criterion.py UT-MOTR/models/
```

#### Step 2: Modify MOTR Model

Edit `UT-MOTR/models/motr.py`:

```python
from models.unitrack_criterion import Unitrackrion

class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                 aux_loss=False, use_unitrack=False, unitrack_weight=1.5):
        super().__init__()
        # ... existing initialization ...
        
        # UniTrack integration
        self.use_unitrack = use_unitrack
        if use_unitrack:
            self.unitrack_criterion = Unitrackrion(
                img_size=(1920, 1080),
                iou_threshold=0.5
            )
            self.unitrack_weight = unitrack_weight
    
    def forward(self, samples, targets=None):
        # ... existing forward pass ...
        
        if self.training:
            # ... existing loss computation ...
            
            # Add UniTrack loss
            if self.use_unitrack:
                # MOTR already has track_ids in outputs
                unitrack_losses = self.unitrack_criterion(outputs, targets)
                for k, v in unitrack_losses.items():
                    losses[k] = self.unitrack_weight * v
            
            return losses
        else:
            return outputs
```

#### Step 3: Update Training Script

Edit `UT-MOTR/main.py`:

```python
def build_model(args):
    # ... existing model building ...
    
    model = MOTR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        use_unitrack=args.use_unitrack,
        unitrack_weight=args.unitrack_weight
    )
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # ... existing arguments ...
    
    # UniTrack arguments
    parser.add_argument('--use_unitrack', action='store_true',
                        help='Enable UniTrack loss')
    parser.add_argument('--unitrack_weight', default=1.5, type=float,
                        help='Weight for UniTrack loss')
    return parser
```

#### Step 4: Training

```bash
cd UT-MOTR
python main.py \
  --output_dir output/motr_unitrack \
  --dataset_file mot \
  --coco_path /path/to/MOT17 \
  --use_unitrack \
  --unitrack_weight 1.5 \
  --batch_size 4 \
  --epochs 50
```

---

## 3. TrackFormer

**Framework Type**: Transformer-based with track queries  
**Integration Difficulty**: Easy  
**Location**: `UT-Trackformer/`

### Integration Steps

#### Step 1: Add the Loss Module

```bash
cp unitrack_criterion.py UT-Trackformer/src/trackformer/
```

#### Step 2: Create Combined Criterion

Edit `UT-Trackformer/src/trackformer/combined_tracking_criterion.py`:

```python
from trackformer.unitrack_criterion import Unitrackrion

class CombinedTrackingCriterion(nn.Module):
    def __init__(self, weight_dict, losses, use_unitrack=False, unitrack_weight=1.5):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        
        # UniTrack integration
        self.use_unitrack = use_unitrack
        if use_unitrack:
            self.unitrack_criterion = Unitrackrion(
                img_size=(1920, 1080),
                iou_threshold=0.5
            )
            self.unitrack_weight = unitrack_weight
    
    def forward(self, outputs, targets):
        # Compute existing losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        
        # Add UniTrack loss
        if self.use_unitrack:
            unitrack_losses = self.unitrack_criterion(outputs, targets)
            for k, v in unitrack_losses.items():
                losses[k] = self.unitrack_weight * v
        
        return losses
```

#### Step 3: Configuration

Edit `UT-Trackformer/cfgs/train.yaml`:

```yaml
model:
  tracking: true
  track_query_false_positive_prob: 0.1
  
loss:
  use_unitrack: true
  unitrack_weight: 1.5
  
training:
  batch_size: 4
  epochs: 50
  lr: 0.0001
```

#### Step 4: Training

```bash
cd UT-Trackformer
python src/train.py \
  --config_path cfgs/train.yaml \
  --output_dir output/trackformer_unitrack
```

---

## 4. ByteTrack

**Framework Type**: Detection-based with byte association  
**Integration Difficulty**: Medium  
**Location**: `UT-BYTE/`

### Integration Steps

#### Step 1: Add the Loss Module

```bash
cp unitrack_criterion.py UT-BYTE/UT-ByteTrack/yolox/models/
```

#### Step 2: Modify YOLOX Head

Edit `UT-BYTE/UT-ByteTrack/yolox/models/yolo_head.py`:

```python
from yolox.models.unitrack_criterion import Unitrackrion

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], 
                 in_channels=[256, 512, 1024], act="silu", 
                 use_unitrack=False, unitrack_weight=1.5):
        super().__init__()
        # ... existing initialization ...
        
        # UniTrack integration
        self.use_unitrack = use_unitrack
        if use_unitrack:
            self.unitrack_criterion = Unitrackrion(
                img_size=(1920, 1080),
                iou_threshold=0.5
            )
            self.unitrack_weight = unitrack_weight
    
    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, 
                   outputs, origin_preds, dtype, track_instances=None):
        # ... existing loss computation ...
        
        # Add UniTrack loss if track instances are provided
        if self.use_unitrack and track_instances is not None:
            # Format outputs for UniTrack
            unitrack_outputs = {
                'pred_boxes': track_instances['pred_boxes'],
                'pred_logits': track_instances['pred_logits'],
                'track_ids': track_instances['track_ids']
            }
            
            # Format targets
            unitrack_targets = self._format_targets_for_unitrack(labels)
            
            # Compute UniTrack loss
            unitrack_losses = self.unitrack_criterion(unitrack_outputs, unitrack_targets)
            
            # Add to total losses
            for k, v in unitrack_losses.items():
                loss_dict[k] = self.unitrack_weight * v
        
        return loss_dict
```

#### Step 3: Training Script

Create `UT-BYTE/UT-ByteTrack/train_sportsmot_unitrack.py`:

```python
import argparse
from yolox.exp import get_exp

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    # ... existing arguments ...
    
    # UniTrack arguments
    parser.add_argument("--use_unitrack", action="store_true",
                        help="Enable UniTrack loss")
    parser.add_argument("--unitrack_weight", type=float, default=1.5,
                        help="Weight for UniTrack loss")
    return parser

def main(exp, args):
    # ... existing training setup ...
    
    # Enable UniTrack in model
    if args.use_unitrack:
        model.head.use_unitrack = True
        model.head.unitrack_weight = args.unitrack_weight
    
    # ... rest of training ...
```

#### Step 4: Training

```bash
cd UT-BYTE/UT-ByteTrack
python train_sportsmot_unitrack.py \
  -f exps/example/mot/yolox_x_mix_mot20_ch.py \
  -d 4 -b 16 \
  --use_unitrack \
  --unitrack_weight 1.5 \
  -o output/bytetrack_unitrack
```

---

## 5. FairMOT

**Framework Type**: One-shot detection and re-ID  
**Integration Difficulty**: Easy  
**Location**: `UT-FairMOT/`

### Integration Steps

#### Step 1: Add the Loss Module

```bash
cp unitrack_criterion.py UT-FairMOT/src/
```

#### Step 2: Modify Trainer

Edit `UT-FairMOT/src/lib/trains/mot.py`:

```python
from src.unitrack_criterion import UTCriterion

class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)
        
        # UniTrack integration
        if opt.use_unitrack:
            self.unitrack_criterion = UTCriterion(
                img_size=(opt.input_w, opt.input_h),
                iou_threshold=opt.unitrack_iou_threshold
            )
            self.unitrack_weight = opt.unitrack_weight
    
    def _get_losses(self, opt):
        # ... existing losses ...
        
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        if opt.use_unitrack:
            loss_states.extend(['unitrack_loss', 'unitrack_tracking', 
                               'unitrack_spatial', 'unitrack_temporal'])
        return loss_states
    
    def run_epoch(self, phase, epoch, data_loader):
        # ... existing training loop ...
        
        for batch in data_loader:
            # ... existing forward pass ...
            
            # Compute UniTrack loss
            if self.opt.use_unitrack:
                unitrack_outputs = {
                    'pred_boxes': output['boxes'],
                    'pred_logits': output['hm'],
                    'track_ids': output['track_ids']
                }
                unitrack_losses = self.unitrack_criterion(unitrack_outputs, batch['targets'])
                
                loss += self.unitrack_weight * unitrack_losses['loss_unitrack']
                loss_stats['unitrack_loss'] = unitrack_losses['loss_unitrack'].item()
```

#### Step 3: Add Options

Edit `UT-FairMOT/src/lib/opts.py`:

```python
def add_unitrack_args(parser):
    parser.add_argument('--use_unitrack', action='store_true',
                        help='Enable UniTrack loss')
    parser.add_argument('--unitrack_weight', type=float, default=1.5,
                        help='Weight for UniTrack loss')
    parser.add_argument('--unitrack_iou_threshold', type=float, default=0.5,
                        help='IoU threshold for UniTrack')
    return parser
```

#### Step 4: Training

```bash
cd UT-FairMOT/src
python train.py mot \
  --exp_id fairmot_unitrack \
  --data_cfg ../src/lib/cfg/mot17.json \
  --use_unitrack \
  --unitrack_weight 1.5 \
  --batch_size 4 \
  --num_epochs 30
```

---

## 6. MOTE

**Framework Type**: Transformer with efficient attention  
**Integration Difficulty**: Easy  
**Location**: `UT-MOTE/`

### Integration Steps

Similar to MOTR with minor modifications for efficient attention mechanism. See `UT-MOTE/models/` for complete implementation.

---

## 7. MOTR-SIG

**Framework Type**: MOTR with spatial-temporal graph  
**Integration Difficulty**: Medium  
**Location**: `UT-MOTR-sig/`

### Integration Steps

Combines MOTR's transformer architecture with graph-based reasoning. UniTrack loss is applied after graph propagation. See `UT-MOTR-sig/src/trackformer/` for complete implementation.

---

## Common Integration Patterns

### Pattern 1: Direct Integration (MOTR, TrackFormer)

For frameworks that already output track IDs:

```python
# Simply add the criterion and compute loss
self.unitrack_criterion = Unitrackrion(img_size, iou_threshold)
unitrack_losses = self.unitrack_criterion(outputs, targets)
total_loss += unitrack_weight * unitrack_losses['loss_unitrack']
```

### Pattern 2: Track ID Extraction (GTR, ByteTrack)

For frameworks that need track ID extraction from association matrices:

```python
# Extract track IDs from association scores
pred_track_ids = self._extract_track_ids_from_associations(asso_matrix)

# Format outputs
unitrack_outputs = {
    'pred_boxes': pred_boxes,
    'pred_logits': pred_logits,
    'track_ids': pred_track_ids
}

# Compute loss
unitrack_losses = self.unitrack_criterion(unitrack_outputs, targets)
```

### Pattern 3: Post-Processing Integration (FairMOT)

For one-shot frameworks:

```python
# After detection and re-ID feature extraction
track_ids = self.tracker.update(detections, reid_features)

# Format for UniTrack
unitrack_outputs = {
    'pred_boxes': detections,
    'pred_logits': class_scores,
    'track_ids': track_ids
}

# Compute loss
unitrack_losses = self.unitrack_criterion(unitrack_outputs, targets)
```

---

## Performance Comparison

| Framework | Baseline IDF1 | +UniTrack IDF1 | Baseline IDS | +UniTrack IDS | Improvement |
|-----------|---------------|----------------|--------------|---------------|-------------|
| GTR | 73.5 | 75.8 | 2829 | 2401 | +2.3% / -15% |
| MOTR | 72.3 | 74.1 | 2439 | 2146 | +1.8% / -12% |
| TrackFormer | 71.8 | 73.9 | 2567 | 2208 | +2.1% / -14% |
| ByteTrack | 77.3 | 78.8 | 2196 | 1976 | +1.5% / -10% |
| FairMOT | 72.5 | 74.4 | 2890 | 2514 | +1.9% / -13% |
| MOTE | 73.1 | 75.1 | 2678 | 2383 | +2.0% / -11% |
| MOTR-SIG | 73.9 | 75.6 | 2345 | 2134 | +1.7% / -9% |

---

## Tips for New Integrations

1. **Start Simple**: Begin with the standalone loss and minimal integration
2. **Verify Track IDs**: Ensure track IDs are properly propagated through your model
3. **Monitor Components**: Watch individual loss components to understand behavior
4. **Tune Gradually**: Start with low weight (0.5) and increase based on validation metrics
5. **Check Format**: Ensure boxes are in [x1, y1, x2, y2] format
6. **Validate Targets**: Confirm ground truth track IDs are correctly formatted

---

For more details on each integration, please refer to the respective framework directories in this repository.
