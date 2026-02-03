#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Test script to verify UniTrack integration with ByteTrack

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unitrack_criterion():
    """Test the Unitrack criterion can be imported and initialized"""
    print("Testing Unitrack criterion...")
    
    try:
        from yolox.models.unitrack_criterion import Unitrackrion
        
        # Initialize the criterion
        criterion = Unitrackrion(img_size=(1920, 1080), iou_threshold=0.5)
        
        # Test forward pass with dummy data
        batch_size = 2
        num_queries = 100
        num_classes = 1
        
        # Create dummy outputs
        outputs = {
            "pred_boxes": torch.randn(batch_size, num_queries, 4),
            "pred_logits": torch.randn(batch_size, num_queries, num_classes),
            "track_ids": torch.randint(0, 10, (batch_size, num_queries)).float()
        }
        
        # Create dummy targets
        targets = []
        for i in range(batch_size):
            targets.append({
                "boxes": torch.randn(5, 4),  # 5 objects
                "labels": torch.randint(0, num_classes, (5,)).float(),
                "track_ids": torch.randint(1, 10, (5,)).float()
            })
        
        # Test forward pass
        losses = criterion(outputs, targets)
        
        print(f"✓ Unitrack criterion works correctly")
        print(f"  - loss_unitrack: {losses['loss_unitrack'].item():.4f}")
        print(f"  - loss_unitrack_tracking: {losses['loss_unitrack_tracking'].item():.4f}")
        print(f"  - loss_unitrack_spatial: {losses['loss_unitrack_spatial'].item():.4f}")
        print(f"  - loss_unitrack_temporal: {losses['loss_unitrack_temporal'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Unitrack criterion test failed: {e}")
        return False

def test_yolox_head():
    """Test the modified YOLOX head with UniTrack"""
    print("\nTesting YOLOX head with UniTrack...")
    
    try:
        from yolox.models.yolo_head import YOLOXHead
        
        # Initialize head with UniTrack enabled
        head = YOLOXHead(
            num_classes=1,
            width=1.0,
            enable_unitrack=True
        )
        
        print(f"✓ YOLOXHead with UniTrack initialized successfully")
        print(f"  - enable_unitrack: {head.enable_unitrack}")
        print(f"  - has unitrack_criterion: {hasattr(head, 'unitrack_criterion')}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLOXHead test failed: {e}")
        return False

def test_yolox_model():
    """Test the YOLOX model with UniTrack head"""
    print("\nTesting YOLOX model with UniTrack...")
    
    try:
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        
        # Create model with UniTrack
        backbone = YOLOPAFPN(depth=1.0, width=1.0)
        head = YOLOXHead(num_classes=1, width=1.0, enable_unitrack=True)
        model = YOLOX(backbone, head)
        
        # Test forward pass
        model.train()
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        # Create dummy targets with track IDs
        targets = torch.randn(batch_size, 50, 6)  # 50 max objects, 6 values (class, cx, cy, w, h, track_id)
        
        # Forward pass
        outputs = model(input_tensor, targets)
        
        print(f"✓ YOLOX model with UniTrack works correctly")
        print(f"  - Output keys: {list(outputs.keys())}")
        print(f"  - Has unitrack_loss: {'unitrack_loss' in outputs}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLOX model test failed: {e}")
        return False

def test_dataset_path():
    """Test if SportsMOT dataset path exists"""
    print("\nTesting SportsMOT dataset path...")
    
    dataset_path = "DATASET_ROOT/sportsmot"
    
    if os.path.exists(dataset_path):
        print(f"✓ SportsMOT dataset found at: {dataset_path}")
        
        # Check for annotation files
        annotations_path = os.path.join(dataset_path, "annotations")
        if os.path.exists(annotations_path):
            json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
            print(f"  - Found {len(json_files)} annotation files: {json_files}")
        
        return True
    else:
        print(f"✗ SportsMOT dataset not found at: {dataset_path}")
        return False

def test_experiment_config():
    """Test if experiment configuration exists"""
    print("\nTesting experiment configuration...")
    
    config_path = "exps/example/mot/yolox_x_sportsmot.py"
    
    if os.path.exists(config_path):
        print(f"✓ Experiment configuration found: {config_path}")
        return True
    else:
        print(f"✗ Experiment configuration not found: {config_path}")
        return False

def main():
    """Run all tests"""
    print("UniTrack Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_unitrack_criterion,
        test_yolox_head,
        test_yolox_model,
        test_dataset_path,
        test_experiment_config,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! UniTrack integration is ready.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 