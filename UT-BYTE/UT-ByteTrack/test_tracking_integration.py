#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test script for tracking integration in training loop
"""

import os
import sys
import torch
import numpy as np
from loguru import logger

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolox.exp import get_exp
from yolox.core.trainer import Trainer
from yolox.tracker.byte_tracker import BYTETracker
import argparse


def test_tracking_integration():
    """Test if tracking integration works correctly"""
    
    logger.info("Testing tracking integration...")
    
    # Create a simple test experiment
    exp = get_exp("exps/example/mot/yolox_x_sportsmot.py", None)
    
    # Create dummy args
    args = argparse.Namespace(
        fp16=False,
        local_rank=0,
        batch_size=2,
        occupy=False,
        resume=False,
        ckpt=None,
        start_epoch=None,
        experiment_name="test_tracking",
        logger="tensorboard"
    )
    
    # Test ByteTracker initialization
    logger.info("Testing ByteTracker initialization...")
    tracking_args = type('Args', (), {
        'track_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'mot20': False,
        'frame_rate': 30
    })()
    
    tracker = BYTETracker(tracking_args)
    logger.info("✓ ByteTracker initialized successfully")
    
    # Test detection format
    logger.info("Testing detection format...")
    dummy_detections = torch.tensor([
        [100, 100, 200, 200, 0.8, 0],  # [x1, y1, x2, y2, score, class]
        [150, 150, 250, 250, 0.9, 0],
        [50, 50, 150, 150, 0.7, 0],
    ], dtype=torch.float32)
    
    img_info = (800, 1440)
    input_size = (800, 1440)
    
    # Test tracking update
    track_results = tracker.update(dummy_detections, img_info, input_size)
    logger.info(f"✓ Tracking update successful: {len(track_results)} tracks")
    
    # Test target tensor format
    logger.info("Testing target tensor format...")
    batch_size = 2
    max_labels = 50
    
    # Create dummy targets [batch_size, max_labels, 6] where 6 = [x, y, w, h, class, track_id]
    targets = torch.zeros((batch_size, max_labels, 6))
    
    # Add some dummy objects
    targets[0, 0] = torch.tensor([0.3, 0.4, 0.2, 0.3, 0, 1])  # class 0, track_id 1
    targets[0, 1] = torch.tensor([0.6, 0.5, 0.15, 0.25, 0, 2])  # class 0, track_id 2
    targets[1, 0] = torch.tensor([0.4, 0.3, 0.3, 0.4, 0, 3])  # class 0, track_id 3
    
    logger.info(f"✓ Target tensor shape: {targets.shape}")
    logger.info(f"✓ Sample targets:\n{targets[0, :2]}")
    
    # Test Unitrack loss with tracking
    logger.info("Testing Unitrack loss with tracking...")
    
    try:
        from yolox.models.unitrack_criterion import Unitrackrion
        
        unitrack_criterion = Unitrackrion(img_size=(800, 1440), iou_threshold=0.5)
        
        # Create dummy outputs for Unitrack loss
        outputs = {
            "pred_boxes": torch.randn(batch_size, 100, 4),
            "pred_logits": torch.randn(batch_size, 100, 1),
            "track_ids": torch.randint(1, 10, (batch_size, 100)).float()
        }
        
        # Create dummy targets for Unitrack loss
        unitrack_targets = []
        for i in range(batch_size):
            unitrack_targets.append({
                "boxes": torch.randn(3, 4),
                "labels": torch.zeros(3),
                "track_ids": torch.tensor([1, 2, 3]).float()
            })
        
        losses = unitrack_criterion(outputs, unitrack_targets)
        logger.info(f"✓ Unitrack loss computation successful: {losses['loss_unitrack'].item():.4f}")
        
    except Exception as e:
        logger.error(f"✗ Unitrack loss test failed: {e}")
        return False
    
    logger.info("✓ All tracking integration tests passed!")
    return True


def test_model_inference():
    """Test if model inference works correctly"""
    
    logger.info("Testing model inference...")
    
    try:
        # Load experiment
        exp = get_exp("exps/example/mot/yolox_x_sportsmot.py", None)
        
        # Create model
        model = exp.get_model()
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 800, 1440)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            
        logger.info(f"✓ Model inference successful, output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'tensor'}")
        
        # Test postprocessing
        from yolox.utils import postprocess
        
        detections = postprocess(outputs, num_classes=1, conf_thre=0.1, nms_thre=0.7)
        logger.info(f"✓ Postprocessing successful: {len(detections)} batch results")
        
        if detections[0] is not None:
            logger.info(f"✓ Detection shape: {detections[0].shape}")
        else:
            logger.info("✓ No detections (expected for random input)")
            
    except Exception as e:
        logger.error(f"✗ Model inference test failed: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    
    logger.info("Starting UT-ByteTrack integration tests...")
    
    # Test 1: Tracking integration
    if not test_tracking_integration():
        logger.error("❌ Tracking integration test failed!")
        return False
    
    # Test 2: Model inference 
    if not test_model_inference():
        logger.error("❌ Model inference test failed!")
        return False
    
    logger.info("🎉 All integration tests passed successfully!")
    logger.info("The tracking integration is ready for training!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 