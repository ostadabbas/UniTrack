import pandas as pd
import numpy as np
from UTloss import Unitrack, UnitrackLoss

def create_test_scenario(scenario_type):
    """Create test data for different tracking scenarios."""
    if scenario_type == "perfect":
        # Perfect tracking: consistent bounding boxes, perfect ID tracking
        preds = []
        gt_frames = []
        gt_ids = []
        for frame in range(1, 11):  # 10 frames
            # Two objects moving smoothly
            preds.extend([
                {'frame': frame, 'id': 1, 'bb_left': 100 + frame*10, 'bb_top': 120, 
                 'bb_width': 60, 'bb_height': 60, 'conf': 0.95, 'x': 0, 'y': 0},
                {'frame': frame, 'id': 2, 'bb_left': 300 + frame*10, 'bb_top': 320, 
                 'bb_width': 60, 'bb_height': 60, 'conf': 0.95, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
        gt_data = pd.DataFrame({
            "X": [100 + f*10 for f in range(10)]*2,  # Smooth motion
            "Y": [120]*10 + [320]*10,
            "Width": [60]*20,
            "Height": [60]*20,
            "Confidence": [1.0]*20
        }, index=pd.MultiIndex.from_arrays([gt_frames, gt_ids], names=("FrameId", "Id")))
        
    elif scenario_type == "medium":
        # Medium tracking: some jitter, occasional low confidence
        preds = []
        gt_frames = []
        gt_ids = []
        for frame in range(1, 11):
            # Add random jitter to positions
            jitter = np.random.normal(0, 5, 4)  # Small position noise
            conf = 0.7 + 0.2 * np.random.random()  # Variable confidence
            
            preds.extend([
                {'frame': frame, 'id': 1, 
                 'bb_left': 100 + frame*10 + jitter[0], 
                 'bb_top': 120 + jitter[1], 
                 'bb_width': 60 + jitter[2], 
                 'bb_height': 60 + jitter[3], 
                 'conf': conf, 'x': 0, 'y': 0},
                {'frame': frame, 'id': 2, 
                 'bb_left': 300 + frame*10 + jitter[0], 
                 'bb_top': 320 + jitter[1], 
                 'bb_width': 60 + jitter[2], 
                 'bb_height': 60 + jitter[3], 
                 'conf': conf, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
        gt_data = pd.DataFrame({
            "X": [100 + f*10 for f in range(10)]*2,
            "Y": [120]*10 + [320]*10,
            "Width": [60]*20,
            "Height": [60]*20,
            "Confidence": [1.0]*20
        }, index=pd.MultiIndex.from_arrays([gt_frames, gt_ids], names=("FrameId", "Id")))
        
    else:  # bad
        # Bad tracking: ID switches, missing detections, wrong sizes
        preds = []
        gt_frames = []
        gt_ids = []
        for frame in range(1, 11):
            if frame % 3 != 0:  # Skip every 3rd frame to simulate missing detections
                if frame == 5:  # Simulate ID switch
                    id1, id2 = 2, 1
                else:
                    id1, id2 = 1, 2
                    
                # Add large random variations
                noise = np.random.normal(0, 30, 4)  # Large position/size noise
                conf = 0.3 + 0.4 * np.random.random()  # Low confidence
                
                preds.extend([
                    {'frame': frame, 'id': id1, 
                     'bb_left': 100 + frame*10 + noise[0], 
                     'bb_top': 120 + noise[1], 
                     'bb_width': 60 + noise[2], 
                     'bb_height': 60 + noise[3], 
                     'conf': conf, 'x': 0, 'y': 0},
                    {'frame': frame, 'id': id2, 
                     'bb_left': 300 + frame*10 + noise[0], 
                     'bb_top': 320 + noise[1], 
                     'bb_width': 60 + noise[2], 
                     'bb_height': 60 + noise[3], 
                     'conf': conf, 'x': 0, 'y': 0}
                ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
        gt_data = pd.DataFrame({
            "X": [100 + f*10 for f in range(10)]*2,
            "Y": [120]*10 + [320]*10,
            "Width": [60]*20,
            "Height": [60]*20,
            "Confidence": [1.0]*20
        }, index=pd.MultiIndex.from_arrays([gt_frames, gt_ids], names=("FrameId", "Id")))
    
    return preds, gt_data

def test_scenarios():
    scenarios = ["perfect", "medium", "bad"]
    results = {}
    
    print("Testing Unitrack metric across different tracking scenarios:")
    print("-" * 60)
    
    for scenario in scenarios:
        preds, gt_data = create_test_scenario(scenario)
        
        # Test Unitrack metric
        unitrack_metric = Unitrack(preds=preds, gt_df=gt_data)
        metric_results = unitrack_metric.compute()
        
        # Test Unitrack loss
        unitrack_loss = UnitrackLoss(preds=preds, gt_df=gt_data)
        loss_value = unitrack_loss.compute_loss()
        
        results[scenario] = {
            'metric': metric_results,
            'loss': loss_value
        }
        
        print(f"\nScenario: {scenario.upper()}")
        print(f"Unitrack Score: {metric_results['unitrack']:.2f}/100")
        print(f"Unitrack Loss: {loss_value:.4f}")
        print("\nComponent Breakdown:")
        print(f"- Tracking Quality: {metric_results['components']['tracking_quality']:.2f}/100")
        print(f"- Spatial Quality: {metric_results['components']['spatial_quality']:.2f}/100")
        print(f"- Temporal Quality: {metric_results['components']['temporal_quality']:.2f}/100")
        print(f"- Avg ID Switches: {metric_results['components']['avg_id_switches']:.2f}")
        print(f"- Avg Fragmentations: {metric_results['components']['avg_fragmentations']:.2f}")
    
    # Validate the results
    validate_results(results)

def validate_results(results):
    """Validate that the metrics behave as expected."""
    print("\nValidating Results:")
    print("-" * 60)
    
    # Check if perfect scenario has highest score
    assert results['perfect']['metric']['unitrack'] > results['medium']['metric']['unitrack'], \
        "Perfect scenario should have higher Unitrack score than medium"
    print("✓ Perfect scenario has higher score than medium")
    
    assert results['medium']['metric']['unitrack'] > results['bad']['metric']['unitrack'], \
        "Medium scenario should have higher Unitrack score than bad"
    print("✓ Medium scenario has higher score than bad")
    
    # Check if bad scenario has highest loss
    assert results['bad']['loss'] > results['medium']['loss'], \
        "Bad scenario should have higher loss than medium"
    print("✓ Bad scenario has higher loss than medium")
    
    assert results['medium']['loss'] > results['perfect']['loss'], \
        "Medium scenario should have higher loss than perfect"
    print("✓ Medium scenario has higher loss than perfect")
    
    # Check component relationships
    perfect_comps = results['perfect']['metric']['components']
    bad_comps = results['bad']['metric']['components']
    
    assert perfect_comps['avg_id_switches'] < bad_comps['avg_id_switches'], \
        "Perfect scenario should have fewer ID switches"
    print("✓ Perfect scenario has fewer ID switches")
    
    assert perfect_comps['temporal_quality'] > bad_comps['temporal_quality'], \
        "Perfect scenario should have better temporal quality"
    print("✓ Perfect scenario has better temporal quality")
    
    print("\nAll validation checks passed! The Unitrack metric is behaving as expected.")

if __name__ == "__main__":
    test_scenarios()
