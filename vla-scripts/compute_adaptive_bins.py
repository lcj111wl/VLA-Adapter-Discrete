"""
compute_adaptive_bins.py

Script to compute adaptive binning statistics for discrete action discretization.
Calculates Min/Max with a margin from RLDS datasets.
"""

import json
import numpy as np
import draccus
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
import dlimp as dl

# Import necessary functions from existing modules
# We need to add the project root to sys.path
import sys
import os
sys.path.append(os.getcwd())

from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.constants import NormalizationType

@dataclass
class BinningConfig:
    data_root_dir: str = "datasets/rlds"
    dataset_name: str = "aloha_scoop_x_into_bowl" 
    output_path: str = "dataset_statistics_256_bins.json"
    n_bins: int = 256
    margin: float = 0.05 # 5% on each side = 10% total
    
@draccus.wrap()
def main(cfg: BinningConfig):
    print(f"Computing stats for {cfg.dataset_name}...")
    
    # 1. Construct Dataset to get Stats
    # We don't need complex transforms, just the raw data to compute stats
    # make_dataset_from_rlds will compute/load stats internally
    
    # Define dummy keys to satisfy requirements
    # These keys depend on the specific dataset (e.g. aloha vs libero)
    # But get_dataset_statistics usually runs on the raw dataset before keys are mapped?
    # Let's look at data_utils.py: get_dataset_statistics iterates over the dataset and keys "action"
    
    # We can try to call make_dataset_from_rlds with minimal args
    # It returns (dataset, dataset_statistics)
    
    # Note: We need to know the robot type to set keys, but let's try generic ones or catch errors
    # For now, let's assume standard keys or try to infer.
    # But simpler: make_dataset_from_rlds returns `dataset_statistics` as the second return value.
    
    # We need to provide *some* keys, otherwise restructure fails.
    # Let's look at what finetune.py uses. It uses RLDSDataset which uses default configs per robot.
    # For simplicity, we will assume the dataset has standard 'action' key which is universal.
    
    # Actually, the cleanest way is to utilize the existing stats file if it exists
    # because iterating TFDS can be slow.
    
    existing_stats_path = Path(cfg.data_root_dir) / cfg.dataset_name / "dataset_statistics.json"
    
    stats = None
    
    # Try finding existing stats first
    # Check various locations
    possible_paths = [
        existing_stats_path,
        Path(f"~/.cache/orca/dataset_statistics_{cfg.dataset_name}.json").expanduser() 
    ]
    
    # If we can't find it, we have to compute it.
    # To compute it, we need to call make_dataset_from_rlds.
    
    # Let's try to perform a "Dry Run" of make_dataset_from_rlds
    try:
        _, stats = make_dataset_from_rlds(
            name=cfg.dataset_name,
            data_dir=cfg.data_root_dir,
            train=True,
            image_obs_keys={"primary": "image"}, # Guessing keys
            depth_obs_keys={},
            state_obs_keys=[],
            action_proprio_normalization_type=NormalizationType.BOUNDS,
        )
    except Exception as e:
        print(f"Could not automatically compute stats via minimal load: {e}")
        print("Please ensure you have run finetune.py at least once to generate dataset_statistics.json, or provide the correct keys.")
        return

    if stats is None:
        print("Failed to get statistics.")
        return

    # 2. Process Stats
    print("Statistics retrieved. Computing adaptive bins...")
    
    # Stats structure: {'action': {'min': [...], 'max': [...]}, ...}
    action_stats = stats['action']
    
    min_vals = np.array(action_stats['min'])
    max_vals = np.array(action_stats['max'])
    
    print(f"Original Min: {min_vals}")
    print(f"Original Max: {max_vals}")
    
    # Calculate New Range with Margin
    range_vals = max_vals - min_vals
    
    # Handle dimensions with 0 range (avoid margin on 0, or keep it 0?)
    # If range is 0, it means that action dim is constant. 
    # We should probably add a small epsilon margin to avoid division by zero later.
    range_vals = np.where(range_vals == 0, 1.0, range_vals) 
    
    margin_val = range_vals * cfg.margin
    
    new_min = min_vals - margin_val
    new_max = max_vals + margin_val
    
    print(f"New Min (w/ margin): {new_min}")
    print(f"New Max (w/ margin): {new_max}")
    
    # 3. Save
    new_stats = {
        "action": {
            "min": new_min.tolist(),
            "max": new_max.tolist(),
            "original_min": min_vals.tolist(),
            "original_max": max_vals.tolist(),
            "n_bins": cfg.n_bins,
            "margin": cfg.margin
        }
    }
    
    with open(cfg.output_path, 'w') as f:
        json.dump(new_stats, f, indent=2)
        
    print(f"Saved adaptive binning stats to {cfg.output_path}")

if __name__ == "__main__":
    main()
