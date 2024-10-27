import os
import random
from pathlib import Path
from tqdm import tqdm

def reduce_dataset(data_dir: str, remove_ratio: float = 0.7, seed: int = 42):
    """
    Removes a specified percentage of images from each category in train and test folders.
    
    Args:
        data_dir: Directory containing train and test folders
        remove_ratio: Proportion of images to remove (default: 0.6)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    data_path = Path(data_dir)
    
    # Process both train and test directories
    for split in ['train', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            print(f"Warning: {split} directory not found!")
            continue
            
        print(f"\nProcessing {split} directory...")
        total_removed = 0
        total_remaining = 0
        
        # Process each disease category
        disease_folders = [f for f in split_path.iterdir() if f.is_dir()]
        
        for disease_folder in tqdm(disease_folders, desc=f"Processing {split} categories"):
            # Get all images in the category
            images = [f for f in disease_folder.glob('*') 
                     if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            
            # Calculate number of images to remove
            num_to_remove = int(len(images) * remove_ratio)
            
            # Randomly select images to remove
            images_to_remove = random.sample(images, num_to_remove)
            
            # Remove selected images
            for img in images_to_remove:
                img.unlink()
                
            remaining_images = len(images) - num_to_remove
            
            total_removed += num_to_remove
            total_remaining += remaining_images
            
            print(f"\n{disease_folder.name}:")
            print(f"Original images: {len(images)}")
            print(f"Removed images: {num_to_remove}")
            print(f"Remaining images: {remaining_images}")
        
        print(f"\nSummary for {split}:")
        print(f"Total images removed: {total_removed}")
        print(f"Total images remaining: {total_remaining}")
        print(f"Actual removal ratio: {total_removed/(total_removed + total_remaining):.2%}")

if __name__ == "__main__":
    # Replace this with your data directory path
    data_directory = "data"
    
    # Confirm before proceeding
    print("WARNING: This script will permanently delete images from your dataset!")
    print(f"Data directory: {data_directory}")
    confirmation = input("Are you sure you want to proceed? (yes/no): ")
    
    if confirmation.lower() == 'yes':
        reduce_dataset(data_directory)
        print("\nDataset reduction completed!")
    else:
        print("\nOperation cancelled.")