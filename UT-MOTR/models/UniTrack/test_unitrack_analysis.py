import pandas as pd
import numpy as np
from UTloss import Unitrack, UnitrackLoss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def create_test_scenario(scenario_type, noise_level=0.0):
    """Create test data for different tracking scenarios."""
    preds = []
    gt_frames = []
    gt_ids = []
    
    if scenario_type == "perfect":
        # Perfect tracking with optional noise
        for frame in range(1, 31):  # Extended to 30 frames
            pos_noise = np.random.normal(0, noise_level, 4)
            size_noise = np.random.normal(0, noise_level * 0.5, 2)
            
            preds.extend([
                {'frame': frame, 'id': 1, 
                 'bb_left': 100 + frame*10 + pos_noise[0], 
                 'bb_top': 120 + pos_noise[1], 
                 'bb_width': 60 + size_noise[0], 
                 'bb_height': 60 + size_noise[0], 
                 'conf': 0.95, 'x': 0, 'y': 0},
                {'frame': frame, 'id': 2, 
                 'bb_left': 300 + frame*10 + pos_noise[2], 
                 'bb_top': 320 + pos_noise[3], 
                 'bb_width': 60 + size_noise[1], 
                 'bb_height': 60 + size_noise[1], 
                 'conf': 0.95, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
    elif scenario_type == "id_switch":
        # Tracking with ID switches at specific frames
        switch_frames = {5, 15, 25}  # ID switches at these frames
        for frame in range(1, 31):
            id1, id2 = (2, 1) if frame in switch_frames else (1, 2)
            pos_noise = np.random.normal(0, noise_level, 4)
            
            preds.extend([
                {'frame': frame, 'id': id1, 
                 'bb_left': 100 + frame*10 + pos_noise[0], 
                 'bb_top': 120 + pos_noise[1], 
                 'bb_width': 60, 'bb_height': 60, 
                 'conf': 0.8, 'x': 0, 'y': 0},
                {'frame': frame, 'id': id2, 
                 'bb_left': 300 + frame*10 + pos_noise[2], 
                 'bb_top': 320 + pos_noise[3], 
                 'bb_width': 60, 'bb_height': 60, 
                 'conf': 0.8, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
    elif scenario_type == "fragmentation":
        # Tracking with object fragmentation (size changes)
        for frame in range(1, 31):
            size_change = 30 * np.sin(frame * np.pi / 5)  # Oscillating size
            pos_noise = np.random.normal(0, noise_level, 4)
            
            preds.extend([
                {'frame': frame, 'id': 1, 
                 'bb_left': 100 + frame*10 + pos_noise[0], 
                 'bb_top': 120 + pos_noise[1], 
                 'bb_width': 60 + size_change, 
                 'bb_height': 60 + size_change, 
                 'conf': 0.7, 'x': 0, 'y': 0},
                {'frame': frame, 'id': 2, 
                 'bb_left': 300 + frame*10 + pos_noise[2], 
                 'bb_top': 320 + pos_noise[3], 
                 'bb_width': 60 - size_change, 
                 'bb_height': 60 - size_change, 
                 'conf': 0.7, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
    elif scenario_type == "occlusion":
        # Tracking with occlusions (missing detections)
        for frame in range(1, 31):
            if frame % 5 != 0:  # Skip every 5th frame
                pos_noise = np.random.normal(0, noise_level, 4)
                preds.extend([
                    {'frame': frame, 'id': 1, 
                     'bb_left': 100 + frame*10 + pos_noise[0], 
                     'bb_top': 120 + pos_noise[1], 
                     'bb_width': 60, 'bb_height': 60, 
                     'conf': 0.6, 'x': 0, 'y': 0},
                    {'frame': frame, 'id': 2, 
                     'bb_left': 300 + frame*10 + pos_noise[2], 
                     'bb_top': 320 + pos_noise[3], 
                     'bb_width': 60, 'bb_height': 60, 
                     'conf': 0.6, 'x': 0, 'y': 0}
                ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
            
    else:  # random_motion
        # Tracking with random motion patterns
        pos1 = np.array([100., 120.])
        pos2 = np.array([300., 320.])
        vel1 = np.array([10., 0.])
        vel2 = np.array([10., 0.])
        
        for frame in range(1, 31):
            # Random acceleration
            acc1 = np.random.normal(0, noise_level, 2)
            acc2 = np.random.normal(0, noise_level, 2)
            
            vel1 += acc1
            vel2 += acc2
            pos1 += vel1
            pos2 += vel2
            
            preds.extend([
                {'frame': frame, 'id': 1, 
                 'bb_left': pos1[0], 'bb_top': pos1[1], 
                 'bb_width': 60, 'bb_height': 60, 
                 'conf': 0.7, 'x': 0, 'y': 0},
                {'frame': frame, 'id': 2, 
                 'bb_left': pos2[0], 'bb_top': pos2[1], 
                 'bb_width': 60, 'bb_height': 60, 
                 'conf': 0.7, 'x': 0, 'y': 0}
            ])
            gt_frames.extend([frame, frame])
            gt_ids.extend([1, 2])
    
    # Create ground truth DataFrame
    gt_data = pd.DataFrame({
        "X": [100 + f*10 for f in range(30)]*2,
        "Y": [120]*30 + [320]*30,
        "Width": [60]*60,
        "Height": [60]*60,
        "Confidence": [1.0]*60
    }, index=pd.MultiIndex.from_arrays([gt_frames, gt_ids], names=("FrameId", "Id")))
    
    return preds, gt_data

def analyze_loss_distribution(n_samples=100):
    """Analyze the distribution of Unitrack loss across different scenarios and noise levels."""
    scenarios = ["perfect", "id_switch", "fragmentation", "occlusion", "random_motion"]
    noise_levels = [0, 5, 10, 20, 40]
    
    results = {
        'scenario': [],
        'noise_level': [],
        'unitrack_score': [],
        'loss_value': [],
        'tracking_quality': [],
        'spatial_quality': [],
        'temporal_quality': [],
        'id_switches': [],
        'fragmentations': []
    }
    
    for scenario in scenarios:
        for noise in noise_levels:
            print(f"Processing {scenario} with noise {noise}...")
            for _ in range(n_samples):
                preds, gt_data = create_test_scenario(scenario, noise)
                
                # Compute Unitrack metric
                unitrack_metric = Unitrack(preds=preds, gt_df=gt_data)
                metric_results = unitrack_metric.compute()
                
                # Compute Unitrack loss
                unitrack_loss = UnitrackLoss(preds=preds, gt_df=gt_data)
                loss_value = unitrack_loss.compute_loss()
                
                # Store results
                results['scenario'].append(scenario)
                results['noise_level'].append(noise)
                results['unitrack_score'].append(metric_results['unitrack'])
                results['loss_value'].append(loss_value)
                results['tracking_quality'].append(metric_results['components']['tracking_quality'])
                results['spatial_quality'].append(metric_results['components']['spatial_quality'])
                results['temporal_quality'].append(metric_results['components']['temporal_quality'])
                results['id_switches'].append(metric_results['components']['avg_id_switches'])
                results['fragmentations'].append(metric_results['components']['avg_fragmentations'])
    
    return pd.DataFrame(results)

def plot_analysis(results_df, output_dir='loss_analysis'):
    """Generate plots analyzing the Unitrack loss behavior."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default style
    plt.style.use('default')
    
    # 1. Distribution of loss values across scenarios
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='scenario', y='loss_value', hue='noise_level')
    plt.title('Distribution of Unitrack Loss Across Scenarios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_distribution.png')
    plt.close()
    
    # 2. Correlation between components
    plt.figure(figsize=(10, 8))
    components = ['tracking_quality', 'spatial_quality', 'temporal_quality', 'loss_value']
    corr = results_df[components].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Unitrack Components')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_correlation.png')
    plt.close()
    
    # 3. Loss vs Noise Level
    plt.figure(figsize=(10, 6))
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        mean_loss = scenario_data.groupby('noise_level')['loss_value'].mean()
        plt.plot(mean_loss.index, mean_loss.values, label=scenario, marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('Unitrack Loss')
    plt.title('Unitrack Loss vs Noise Level')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_vs_noise.png')
    plt.close()
    
    # 4. Component breakdown by scenario
    components = ['tracking_quality', 'spatial_quality', 'temporal_quality']
    plt.figure(figsize=(12, 6))
    mean_components = results_df[components + ['scenario']].groupby('scenario').mean()
    mean_components.plot(kind='bar')
    plt.title('Average Component Scores by Scenario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_breakdown.png')
    plt.close()
    
    # 5. ID switches and fragmentations
    plt.figure(figsize=(10, 6))
    metrics = ['id_switches', 'fragmentations']
    mean_metrics = results_df[metrics + ['scenario']].groupby('scenario').mean()
    mean_metrics.plot(kind='bar')
    plt.title('Average Tracking Errors by Scenario')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tracking_errors.png')
    plt.close()
    
    # Normality tests
    normality_results = {}
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]['loss_value']
        stat, p_value = stats.normaltest(scenario_data)
        normality_results[scenario] = {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'mean': scenario_data.mean(),
            'std': scenario_data.std(),
            'skewness': stats.skew(scenario_data),
            'kurtosis': stats.kurtosis(scenario_data)
        }
    
    return normality_results

def main():
    print("Starting Unitrack loss analysis...")
    results_df = analyze_loss_distribution(n_samples=50)
    
    print("\nGenerating plots and computing normality tests...")
    normality_results = plot_analysis(results_df)
    
    print("\nNormality Test Results:")
    print("-" * 60)
    for scenario, results in normality_results.items():
        print(f"\nScenario: {scenario}")
        print(f"Test Statistic: {results['statistic']:.2f}")
        print(f"P-value: {results['p_value']:.4f}")
        print(f"Distribution is {'normal' if results['is_normal'] else 'not normal'}")
    
    # Save results
    results_df.to_csv('loss_analysis/unitrack_analysis_results.csv', index=False)
    print("\nAnalysis complete! Results saved to loss_analysis/")

if __name__ == "__main__":
    main()
