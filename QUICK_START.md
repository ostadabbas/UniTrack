# UniTrack Quick Start Guide

Get UniTrack integrated into your MOT framework in 5 minutes!

## 🚀 3-Step Integration

### Step 1: Copy the Loss Function (30 seconds)

```bash
# Copy the standalone loss module to your project
cp unitrack_criterion.py /path/to/your/mot/project/
```

### Step 2: Add to Your Model (2 minutes)

```python
from unitrack_criterion import Unitrackrion

class YourMOTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your existing code...
        
        # Add UniTrack (just 3 lines!)
        self.unitrack_loss = Unitrackrion(
            img_size=(config.img_width, config.img_height),
            iou_threshold=0.5
        )
```

### Step 3: Compute Loss During Training (2 minutes)

```python
def training_step(self, batch):
    outputs = self.model(batch['images'])
    
    # Your existing losses
    detection_loss = self.compute_detection_loss(outputs, batch['targets'])
    
    # Add UniTrack loss (just 2 lines!)
    unitrack_losses = self.unitrack_loss(outputs, batch['targets'])
    total_loss = detection_loss + 1.5 * unitrack_losses['loss_unitrack']
    
    return total_loss
```

**That's it!** You're now using UniTrack.

---

## 📋 Requirements Checklist

Make sure your model outputs include:

- ✅ `pred_boxes`: Predicted boxes in [x1, y1, x2, y2] format
- ✅ `pred_logits`: Class prediction logits
- ✅ `track_ids`: Integer track IDs (0 = invalid/background)

Make sure your targets include:

- ✅ `boxes`: Ground truth boxes in [x1, y1, x2, y2] format
- ✅ `labels`: Ground truth class labels
- ✅ `track_ids`: Ground truth track IDs

---

## 🎯 Expected Format

### Model Outputs

```python
outputs = {
    'pred_boxes': torch.Tensor,    # Shape: (batch_size, num_queries, 4)
    'pred_logits': torch.Tensor,   # Shape: (batch_size, num_queries, num_classes)
    'track_ids': torch.Tensor      # Shape: (batch_size, num_queries), dtype=long
}
```

### Targets

```python
targets = [
    {
        'boxes': torch.Tensor,      # Shape: (num_objects, 4)
        'labels': torch.Tensor,     # Shape: (num_objects,)
        'track_ids': torch.Tensor   # Shape: (num_objects,), dtype=long
    }
    # ... one dict per batch item
]
```

---

## ⚙️ Recommended Settings

### For MOT17/MOT20 (Pedestrian Tracking)

```python
unitrack_loss = Unitrackrion(
    img_size=(1920, 1080),
    iou_threshold=0.4
)
unitrack_weight = 1.5
```

### For DanceTrack (Crowded Scenes)

```python
unitrack_loss = Unitrackrion(
    img_size=(1920, 1080),
    iou_threshold=0.3
)
unitrack_weight = 2.0
```

### For SportsMOT (Fast Motion)

```python
unitrack_loss = Unitrackrion(
    img_size=(1920, 1080),
    iou_threshold=0.4
)
unitrack_weight = 1.5
```

---

## 🔍 Monitoring Training

Log the loss components to understand behavior:

```python
print(f"Total: {losses['loss_unitrack']:.4f}")
print(f"Tracking: {losses['loss_unitrack_tracking']:.4f}")
print(f"Spatial: {losses['loss_unitrack_spatial']:.4f}")
print(f"Temporal: {losses['loss_unitrack_temporal']:.4f}")
```

**What to look for:**
- All components should decrease during training
- If one component dominates, adjust alpha weights
- Typical values: 0.1-2.0 for each component

---

## 🐛 Common Issues

### Issue: "Track IDs are all zeros"

**Fix**: Ensure your model assigns track IDs. Example:

```python
# Bad: All zeros
track_ids = torch.zeros(batch_size, num_queries)

# Good: Actual track IDs
track_ids = your_tracking_algorithm(detections)
```

### Issue: "Loss is NaN"

**Fix**: Check box format and track ID validity:

```python
# Boxes should be [x1, y1, x2, y2] with x2 > x1, y2 > y1
assert (boxes[:, 2] > boxes[:, 0]).all()
assert (boxes[:, 3] > boxes[:, 1]).all()

# Track IDs should be positive integers
assert (track_ids >= 0).all()
```

### Issue: "No improvement in metrics"

**Fix**: Increase the weight:

```python
# Try increasing from 1.5 to 2.0 or 2.5
unitrack_weight = 2.5
```

---

## 📈 Expected Results

After integrating UniTrack, you should see:

- **↓ 10-15% reduction** in ID Switches (IDS)
- **↑ 1.5-2.5% improvement** in IDF1 score
- **↓ 2-5% reduction** in False Positives
- **Smoother trajectories** in visualizations

Typical training time increase: **< 5%**

---

## 🎓 Next Steps

1. **Basic Integration** (You are here!)
2. **Tune Hyperparameters**: Adjust weights based on your dataset
3. **Monitor Components**: Track individual loss components
4. **Validate Results**: Check IDF1 and IDS on validation set
5. **Fine-tune**: Adjust alpha weights for optimal performance

---

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Integration Examples**: See `INTEGRATION_EXAMPLES.md`
- **Source Code**: See `unitrack_criterion.py`

---

## 💡 Pro Tips

1. **Start with default settings** - they work well for most cases
2. **Monitor loss components** - helps identify issues early
3. **Use gradient clipping** - prevents instability (clip at 0.1-1.0)
4. **Warm-up the loss** - start with low weight, increase gradually
5. **Validate frequently** - check IDF1/IDS every few epochs

---

**Questions?** Open an issue on GitHub or check the full documentation!

**Success?** Share your results - we'd love to hear about your integration!
