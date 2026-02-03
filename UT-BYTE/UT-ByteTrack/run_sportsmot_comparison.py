#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import subprocess
import sys
import time
from datetime import datetime

def run_experiment(exp_name, config_file, enable_unitrack=False):
    """
    Run a single experiment with given configuration
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Config file: {config_file}")
    print(f"UniTrack enabled: {enable_unitrack}")
    print(f"{'='*60}\n")
    
    # Create experiment directory
    exp_dir = f"YOLOX_outputs/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Prepare training command
    cmd = [
        "python", "train_sportsmot_unitrack.py",
        "-f", config_file,
        "-d", "1",  # Use 1 GPU
        "-b", "8",  # Batch size
        "--fp16",  # Use mixed precision
        "-expn", exp_name,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        end_time = time.time()
        
        # Save logs
        log_file = os.path.join(exp_dir, "training_log.txt")
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"End time: {datetime.fromtimestamp(end_time)}\n")
            f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ Experiment {exp_name} completed successfully")
            return True
        else:
            print(f"✗ Experiment {exp_name} failed with return code {result.returncode}")
            print(f"Check log file: {log_file}")
            return False
            
    except Exception as e:
        print(f"✗ Error running experiment {exp_name}: {e}")
        return False

def parse_results(exp_dir):
    """
    Parse training results from experiment directory
    """
    results = {}
    
    # Look for evaluation results
    eval_file = os.path.join(exp_dir, "train_log.txt")
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r") as f:
                content = f.read()
                # Extract AP50 and AP50_95 from logs
                lines = content.split('\n')
                for line in lines:
                    if "Average Precision" in line and "AP50" in line:
                        # Parse AP metrics
                        if "AP50_95" in line:
                            try:
                                ap50_95 = float(line.split(":")[-1].strip())
                                results["AP50_95"] = ap50_95
                            except:
                                pass
                        elif "AP50" in line and "AP50_95" not in line:
                            try:
                                ap50 = float(line.split(":")[-1].strip())
                                results["AP50"] = ap50
                            except:
                                pass
        except Exception as e:
            print(f"Error parsing results from {eval_file}: {e}")
    
    return results

def compare_results(regular_results, unitrack_results):
    """
    Compare results between regular ByteTrack and UT-ByteTrack
    """
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    print(f"{'Metric':<15} {'Regular ByteTrack':<20} {'UT-ByteTrack':<20} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    for metric in ["AP50", "AP50_95"]:
        regular_val = regular_results.get(metric, 0.0)
        unitrack_val = unitrack_results.get(metric, 0.0)
        
        if regular_val > 0:
            improvement = ((unitrack_val - regular_val) / regular_val) * 100
            print(f"{metric:<15} {regular_val:<20.4f} {unitrack_val:<20.4f} {improvement:<15.2f}%")
        else:
            print(f"{metric:<15} {regular_val:<20.4f} {unitrack_val:<20.4f} {'N/A':<15}")
    
    print(f"{'-'*70}")
    
    # Summary
    if unitrack_results.get("AP50", 0) > regular_results.get("AP50", 0):
        print("✓ UT-ByteTrack shows improvement over regular ByteTrack!")
    else:
        print("✗ UT-ByteTrack did not improve over regular ByteTrack")

def main():
    """
    Main function to run both experiments and compare results
    """
    print("SportsMOT ByteTrack vs UT-ByteTrack Comparison")
    print("=" * 50)
    
    # Check if SportsMOT dataset exists
    dataset_path = "DATASET_ROOT/sportsmot"
    if not os.path.exists(dataset_path):
        print(f"Error: SportsMOT dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Experiment configurations
    config_file = "exps/example/mot/yolox_x_sportsmot.py"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)
    
    # Create a version without UniTrack for comparison
    regular_config = config_file.replace("sportsmot.py", "sportsmot_regular.py")
    
    # Create regular ByteTrack config (without UniTrack)
    if not os.path.exists(regular_config):
        print(f"Creating regular ByteTrack config: {regular_config}")
        with open(config_file, "r") as f:
            content = f.read()
        
        # Modify content to disable UniTrack
        regular_content = content.replace("self.enable_unitrack = True", "self.enable_unitrack = False")
        regular_content = regular_content.replace("enable_unitrack=self.enable_unitrack", "enable_unitrack=False")
        
        with open(regular_config, "w") as f:
            f.write(regular_content)
    
    # Run experiments
    experiments = [
        ("ByteTrack_SportsMOT_Regular", regular_config, False),
        ("ByteTrack_SportsMOT_UniTrack", config_file, True),
    ]
    
    results = {}
    
    for exp_name, config, enable_unitrack in experiments:
        success = run_experiment(exp_name, config, enable_unitrack)
        if success:
            exp_dir = f"YOLOX_outputs/{exp_name}"
            results[exp_name] = parse_results(exp_dir)
        else:
            results[exp_name] = {}
    
    # Compare results
    if "ByteTrack_SportsMOT_Regular" in results and "ByteTrack_SportsMOT_UniTrack" in results:
        compare_results(
            results["ByteTrack_SportsMOT_Regular"],
            results["ByteTrack_SportsMOT_UniTrack"]
        )
    
    # Save comparison results
    comparison_file = "sportsmot_comparison_results.txt"
    with open(comparison_file, "w") as f:
        f.write("SportsMOT ByteTrack vs UT-ByteTrack Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        for exp_name, exp_results in results.items():
            f.write(f"{exp_name}:\n")
            for metric, value in exp_results.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
    
    print(f"\nComparison results saved to: {comparison_file}")

if __name__ == "__main__":
    main() 