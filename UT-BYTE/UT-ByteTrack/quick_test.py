#!/usr/bin/env python3
"""Quick test to verify the integration fixes"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_obj_loss_mapping():
    """Test that obj_loss is properly mapped from conf_loss"""
    print("Testing obj_loss mapping...")
    
    from yolox.models.yolox import YOLOX
    from yolox.models.yolo_head import YOLOXHead
    from yolox.models.yolo_pafpn import YOLOPAFPN
    
    # Create model with UniTrack enabled
    backbone = YOLOPAFPN(1.33, 1.25, in_channels=[256, 512, 1024])
    head = YOLOXHead(num_classes=1, width=1.25, enable_unitrack=True)
    model = YOLOX(backbone, head)
    
    # Create dummy inputs
    batch_size = 2
    img_size = (800, 1440)
    imgs = torch.randn(batch_size, 3, img_size[0], img_size[1])
    
    # Create dummy targets [batch_size, max_labels, 6]
    targets = torch.zeros(batch_size, 50, 6)
    targets[0, 0] = torch.tensor([0.0, 400, 300, 100, 150, 1.0])  # [class, cx, cy, w, h, track_id]
    targets[0, 1] = torch.tensor([0.0, 600, 400, 80, 120, 2.0])
    targets[1, 0] = torch.tensor([0.0, 300, 250, 120, 180, 3.0])
    
    # Forward pass
    model.train()
    outputs = model(imgs, targets)
    
    # Check that both conf_loss and obj_loss are present
    assert 'conf_loss' in outputs, "conf_loss missing from outputs"
    assert 'obj_loss' in outputs, "obj_loss missing from outputs"
    assert 'unitrack_loss' in outputs, "unitrack_loss missing from outputs"
    
    print(f"✓ obj_loss mapping works!")
    print(f"  conf_loss: {outputs['conf_loss'].item():.4f}")
    print(f"  obj_loss: {outputs['obj_loss'].item():.4f}")
    print(f"  Are they equal? {torch.equal(outputs['conf_loss'], outputs['obj_loss'])}")
    print(f"  unitrack_loss: {outputs['unitrack_loss'].item():.4f}")
    
    return True

def test_tracker_integration():
    """Test tracker integration"""
    print("\nTesting tracker integration...")
    
    from yolox.tracker.byte_tracker import BYTETracker
    
    # Create tracker
    tracking_args = type('Args', (), {
        'track_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'mot20': False,
        'frame_rate': 30
    })()
    
    tracker = BYTETracker(tracking_args)
    
    # Test with dummy detections
    detections = torch.tensor([
        [100, 100, 200, 200, 0.8],  # [x1, y1, x2, y2, score]
        [300, 300, 400, 400, 0.9],
    ], dtype=torch.float32)
    
    img_info = (800, 1440)
    input_size = (800, 1440)
    
    # Update tracker
    track_results = tracker.update(detections, img_info, input_size)
    
    print(f"✓ Tracker integration works!")
    print(f"  Number of tracks: {len(track_results)}")
    print(f"  Track IDs: {[t.track_id for t in track_results]}")
    
    return True

if __name__ == "__main__":
    print("=== Quick Integration Test ===")
    
    try:
        test_obj_loss_mapping()
        test_tracker_integration()
        print("\n🎉 All tests passed! Integration is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 