import os
import json
import fiftyone as fo
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
from PIL import Image  # For getting image dimensions

def parse_gui_odyssey_dataset(
    root_dir: str,
    split_names: List[str],
    split_file: str = "splits/random_split.json",
    limit_episodes: Optional[int] = None
) -> Dict[str, List[fo.Sample]]:
    """
    Parse the GUI-Odyssey dataset into FiftyOne samples.
    
    Args:
        root_dir: Path to the root directory of the GUI-Odyssey dataset
        split_names: List of splits to process ("train", "test", or both)
        split_file: Path to the split file, relative to root_dir
        limit_episodes: Optional limit on the number of episodes to process per split
        
    Returns:
        Dictionary mapping split names to lists of FiftyOne samples
    """
    # Load split information
    with open(os.path.join(root_dir, split_file), 'r') as f:
        splits = json.load(f)
    
    results = {}
    
    for split_name in split_names:
        if split_name not in splits:
            raise ValueError(f"Split '{split_name}' not found in split file")
        
        # Get annotation filenames for this split
        episode_files = splits[split_name]
        if limit_episodes:
            episode_files = episode_files[:limit_episodes]
        
        print(f"Processing {len(episode_files)} episodes for '{split_name}' split...")
        
        # Process each episode
        samples = []
        for episode_file in tqdm(episode_files):
            episode_samples = process_episode(
                os.path.join(root_dir, "annotations", episode_file),
                root_dir
            )
            samples.extend(episode_samples)
        
        results[split_name] = samples
        print(f"Created {len(samples)} samples for '{split_name}' split")
    
    return results


def process_episode(annotation_path: str, root_dir: str) -> List[fo.Sample]:
    """
    Process a single episode annotation file into FiftyOne samples.
    
    Args:
        annotation_path: Path to the annotation JSON file
        root_dir: Path to the root directory of the GUI-Odyssey dataset
        
    Returns:
        List of FiftyOne samples for this episode
    """
    # Load annotation data
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    episode_id = data["episode_id"]
    device_info = data["device_info"]
    task_info = data["task_info"]
    steps = data["steps"]
    
    # Process each step in the episode
    samples = []
    history = []
    
    for step_data in steps:
        # Create a new sample
        step_number = step_data["step"]
        screenshot_filename = step_data["screenshot"]
        screenshot_path = os.path.join(root_dir, "screenshots", screenshot_filename)
        
        # Skip sample if image doesn't exist or can't be opened
        if not os.path.exists(screenshot_path):
            print(f"Warning: Image file not found: {screenshot_path}. Skipping sample.")
            continue
        
        sample = fo.Sample(filepath=screenshot_path)
        
        # Add basic metadata
        sample["episode_id"] = episode_id
        sample["device_name"] = fo.Classification(label=device_info["device_name"])
        sample["step"] = step_number
        
        # Add task information
        sample["category"] = fo.Classification(label=task_info["category"])
        sample["meta_task"] = fo.Classification(label=task_info["meta_task"])
        sample["task"] = task_info["task"]
        sample["instruction"] = task_info["instruction"]
        
        # Add app classifications
        sample["apps_used"] = fo.Classifications(classifications=[fo.Classification(label=app) for app in task_info["app"]])
        
        # Process action based on type
        action = step_data["action"]
        info = step_data["info"]
        
        # Create structured history
        step_history = {
            "step": step_number,
            "action": action,
            "info": info
        }
        history.append(step_history)
        sample["structured_history"] = history.copy()  # Create a copy to avoid reference issues
        
        # Handle different action types
        if action in ["CLICK", "LONG_PRESS"]:
            # For point-based actions
            if isinstance(info, list):
                # Create a list of Keypoint objects
                keypoints = []
                
                # Format is typically [[x, y], [x, y]] but we only need one point
                x, y = info[0]
                norm_x = (float(x) / 1000.0) 
                norm_y = (float(y) / 1000.0) 
                
                # Create a single Keypoint
                keypoint = fo.Keypoint(
                    label=action,
                    points=[[norm_x, norm_y]]  # Coordinates as [[x, y]]
                )
                keypoints.append(keypoint)
                
                # Add keypoints to the sample
                sample["action_points"] = fo.Keypoints(keypoints=keypoints)
                
            elif isinstance(info, str) and info.startswith("KEY_"):
                # Handle system key actions (e.g., KEY_HOME)
                label = info.replace("KEY_", "")
                sample["action_press"] = fo.Classification(label=label)
                
        elif action == "SCROLL":
            # For scroll actions
            if isinstance(info, list):
                # Format is [[x1, y1], [x2, y2]] (start and end positions)
                start_x, start_y = info[0]
                end_x, end_y = info[1]
                
                # Calculate deltas
                dx = end_x - start_x
                dy = end_y - start_y
                
                # Determine the primary direction based on which delta is larger
                if abs(dx) > abs(dy):
                    # Horizontal scroll is dominant
                    scroll_direction = "LEFT" if dx > 0 else "RIGHT"
                else:
                    # Vertical scroll is dominant
                    scroll_direction = "UP" if dy > 0 else "DOWN"
                
                # Normalize coordinates
                norm_start_x = (float(start_x) / 1000.0) 
                norm_start_y = (float(start_y) / 1000.0) 
                norm_end_x = (float(end_x) / 1000.0) 
                norm_end_y = (float(end_y) / 1000.0) 
                
                # Create a polyline representing the scroll path
                polyline = fo.Polyline(
                    label=f"{action}_{scroll_direction}",
                    points=[[[norm_start_x, norm_start_y], [norm_end_x, norm_end_y]]],
                    closed=True,
                    filled=False
                )
                
                # Add the polyline to a Polylines object
                sample["action_scroll"] = fo.Polylines(polylines=[polyline])
            
        elif action == "TEXT":
            # For text input actions
            sample["action_type"] = fo.Classification(label=info)
        
        elif action in ["COMPLETE", "IMPOSSIBLE", "INCOMPLETE"]:
            # For terminal actions
            sample["action_end"] = fo.Classification(label=action)
        
        elif action in ["HOME", "BACK", "RECENT"]:
            # For navigation actions
            sample["action_press"] = fo.Classification(label=action)
        
        samples.append(sample)
    
    return samples


def create_fiftyone_dataset(
    root_dir: str,
    split_name: str,
    dataset_name: Optional[str] = None,
    limit_episodes: Optional[int] = None
) -> fo.Dataset:
    """
    Create a FiftyOne dataset for the specified split.
    
    Args:
        root_dir: Path to the root directory of the GUI-Odyssey dataset
        split_name: Split to process ("train" or "test")
        dataset_name: Optional name for the FiftyOne dataset
        limit_episodes: Optional limit on the number of episodes to process
        
    Returns:
        FiftyOne dataset
    """
    if dataset_name is None:
        dataset_name = f"gui-odyssey-{split_name}"
    
    # Create a new dataset
    dataset = fo.Dataset(dataset_name, overwrite=True)
    
    # Parse the data
    split_results = parse_gui_odyssey_dataset(
        root_dir=root_dir,
        split_names=[split_name],
        limit_episodes=limit_episodes
    )
    
    # Add the samples to the dataset
    dataset.add_samples(split_results[split_name])
    
    return dataset


# Path to the GUI-Odyssey dataset
ROOT_DIR = "GUI-Odyssey"

# Create training and test datasets
train_dataset = create_fiftyone_dataset(
    root_dir=ROOT_DIR,
    split_name="train",
    dataset_name="gui-odyssey-train",
    limit_episodes=None  # Set a number to limit processing for testing
)

test_dataset = create_fiftyone_dataset(
    root_dir=ROOT_DIR,
    split_name="test",
    dataset_name="gui-odyssey-test",
    limit_episodes=None
)
