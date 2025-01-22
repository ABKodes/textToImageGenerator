import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        dataset_dir (str): Path to the directory containing character folders.
        output_dir (str): Path to the output directory for split datasets.
        train_ratio (float): Ratio of training data (default 0.8).
        val_ratio (float): Ratio of validation data (default 0.1).
        test_ratio (float): Ratio of test data (default 0.1).
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum up to 1"

    # Create output directories for splits
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)

    # Process each character folder
    for char_folder in os.listdir(dataset_dir):
        char_folder_path = os.path.join(dataset_dir, char_folder)

        if not os.path.isdir(char_folder_path):
            continue

        images = [f for f in os.listdir(char_folder_path) if f.endswith('.png')]
        random.shuffle(images)

        # Calculate split sizes
        num_train = int(len(images) * train_ratio)
        num_val = int(len(images) * val_ratio)

        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(char_folder_path, img), os.path.join(train_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(char_folder_path, img), os.path.join(val_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(char_folder_path, img), os.path.join(test_dir, img))

        print(f"Processed '{char_folder}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("Dataset split completed successfully.")

# Example usage
dataset_dir = 'amharic_dataset'  # Original dataset directory
output_dir = 'amharic_dataset_split'  # Output directory for split data
split_dataset(dataset_dir, output_dir)
