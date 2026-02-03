#!/usr/bin/env python
"""
Test script to check model loading with different configurations
"""
import torch
from main import get_args_parser
from models import build_model

def test_model_loading(config_name, num_queries):
    """Test model loading with specific configuration"""
    print(f"\n--- Testing {config_name} with {num_queries} queries ---")
    
    # Create arguments
    parser = get_args_parser()
    args = parser.parse_args([
        '--dataset_file', 'e2e_mot',
        '--meta_arch', 'motr_unincost',
        '--num_queries', str(num_queries),
        '--query_interaction_layer', 'GQIM',
        '--with_box_refine',
        '--g_size', '1'
    ])
    
    try:
        # Build model without pretrained weights
        model, criterion, postprocessors = build_model(args)
        print(f"✓ Model built successfully")
        print(f"  - num_queries: {model.num_queries}")
        print(f"  - query_embed shape: {model.query_embed.weight.shape}")
        print(f"  - position shape: {model.position.weight.shape}")
        
        if hasattr(model, 'position_offset'):
            print(f"  - position_offset shape: {model.position_offset.weight.shape}")
        if hasattr(model, 'query_embed_offset'):
            print(f"  - query_embed_offset shape: {model.query_embed_offset.weight.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        return False

def test_pretrained_loading():
    """Test loading the problematic pretrained model"""
    print(f"\n--- Testing Pretrained Model Loading ---")
    
    pretrained_path = "USER_HOME/aclab/CO-MOT/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
    
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✓ Model state dict found")
            
            # Check key shapes
            for key in ['query_embed.weight', 'class_embed.0.weight', 'class_embed.0.bias']:
                if key in state_dict:
                    print(f"  - {key}: {state_dict[key].shape}")
                else:
                    print(f"  - {key}: NOT FOUND")
            
            # Count classes in class_embed
            if 'class_embed.0.weight' in state_dict:
                num_classes = state_dict['class_embed.0.weight'].shape[0]
                print(f"  - Pretrained model has {num_classes} classes")
                
            # Count queries
            if 'query_embed.weight' in state_dict:
                num_queries_pretrained = state_dict['query_embed.weight'].shape[0]
                print(f"  - Pretrained model has {num_queries_pretrained} queries")
                
        return True
        
    except Exception as e:
        print(f"✗ Failed to load pretrained model: {e}")
        return False

if __name__ == "__main__":
    print("CO-MOT Model Loading Test")
    print("=" * 50)
    
    # Test different configurations
    test_model_loading("60 queries", 60)
    test_model_loading("300 queries", 300)
    
    # Test pretrained model
    test_pretrained_loading()
    
    print("\n" + "=" * 50)
    print("Test completed!") 