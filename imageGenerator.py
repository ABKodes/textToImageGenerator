import os
import json

# Dataset directory
dataset_dir = 'amharic_dataset'

# JSON output file
output_json_file = 'amharic_dataset_labels.json'

# Initialize the dataset list
dataset_labels = []

# Traverse the dataset directory
for char_folder_name in os.listdir(dataset_dir):
    char_folder_path = os.path.join(dataset_dir, char_folder_name)

    # Skip if it's not a directory
    if not os.path.isdir(char_folder_path):
        continue

    # Get all image files in the directory
    image_files = [f for f in os.listdir(char_folder_path) if f.endswith('.png')]

    # Create label entries for each image
    for image_file in image_files:
        image_path = os.path.join(char_folder_path, image_file)
        label_entry = {
            "image_path": image_path,
            "label": char_folder_name
        }
        dataset_labels.append(label_entry)

# Save the dataset labels to a JSON file
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json.dump({"dataset": dataset_labels}, json_file, ensure_ascii=False, indent=4)

print(f"Labels have been saved to {output_json_file}")
