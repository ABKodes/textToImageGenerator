from PIL import Image, ImageDraw, ImageFont
import os

# Path to the text file containing the scraped text
input_file = 'text.txt'

# Root directory for the dataset
dataset_dir = 'amharic_dataset'
os.makedirs(dataset_dir, exist_ok=True)  # Ensure the root directory exists

# Directory containing font files
font_dir = 'fonts'  # Replace with your font directory path
font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf', '.TTF'))]  # List all font files

# File to track processed fonts
processed_fonts_file = 'processed_fonts.txt'

# Create or load the processed fonts list
if os.path.exists(processed_fonts_file):
    with open(processed_fonts_file, 'r', encoding='utf-8') as file:
        processed_fonts = set(file.read().splitlines())
else:
    processed_fonts = set()

# Open the text file and read content
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Loop through each font in the font directory
for font_file in font_files:
    if font_file in processed_fonts:
        print(f"Skipping already processed font: {font_file}")
        continue

    font_path = os.path.join(font_dir, font_file)
    print(f"Processing new font: {font_file}")

    # Load the font
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    # Loop through each character in the text
    for char in text:
        if not char.isspace():  # Skip spaces and newlines
            # Normalize the character folder name
            char_folder_name = f"{char}"  # e.g., 'ሀ' for ሀ
            char_folder_path = os.path.join(dataset_dir, char_folder_name)

            # Create a directory for the character if it doesn't exist
            os.makedirs(char_folder_path, exist_ok=True)

            # Determine the starting index for new images
            existing_files = os.listdir(char_folder_path)
            max_existing_index = max(
                [int(file.split('_')[-1].split('.')[0]) for file in existing_files if file.endswith('.png')],
                default=0
            )
            start_index = max_existing_index + 1

            # Generate 20 new images for this character for the current font
            for i in range(start_index, start_index + 1):
                char_image_name = f"{char}_{i}.png"
                char_image_path = os.path.join(char_folder_path, char_image_name)

                # Create a blank image
                image_size = (32,32)  # Width, Height set to 32x32
                image = Image.new('L', image_size, color='white')  # 'L' mode for grayscale

                # Draw the character on the image
                draw = ImageDraw.Draw(image)
                text_bbox = draw.textbbox((0, 0), char, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = (image_size[0] - text_width) // 2
                text_y = (image_size[1] - text_height) // 2
                draw.text((text_x, text_y), char, fill='black', font=font)

                # Save the image
                image.save(char_image_path)
                print(f"Saved image for '{char}' as {char_image_path}")

    # Mark the font as processed
    processed_fonts.add(font_file)
    with open(processed_fonts_file, 'a', encoding='utf-8') as file:
        file.write(f"{font_file}\n")

# Function to delete images within a specified range and renumber the remaining files
def delete_images_in_range(start_index, end_index):
    print(f"Deleting images from index {start_index} to {end_index} for every character...")

    # Iterate over all character folders in the dataset directory
    for char_folder_name in os.listdir(dataset_dir):
        char_folder_path = os.path.join(dataset_dir, char_folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(char_folder_path):
            continue

        # Get all image files in the directory
        image_files = [f for f in os.listdir(char_folder_path) if f.endswith('.png')]

        # Select images within the specified range
        images_to_delete = [
            f for f in image_files
            if start_index <= int(f.split('_')[-1].split('.')[0]) <= end_index
        ]

        # Delete the selected images
        for image_name in images_to_delete:
            image_path = os.path.join(char_folder_path, image_name)
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except FileNotFoundError:
                print(f"File not found, skipping: {image_path}")

        # Renumber the remaining images
        remaining_files = [f for f in os.listdir(char_folder_path) if f.endswith('.png')]
        remaining_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for new_index, file_name in enumerate(remaining_files, start=1):
            old_path = os.path.join(char_folder_path, file_name)
            new_name = f"{char_folder_name}_{new_index}.png"
            new_path = os.path.join(char_folder_path, new_name)
            os.rename(old_path, new_path)

    print(f"Deletion of images from index {start_index} to {end_index} and renumbering is complete.")

# Ask user if they want to delete images and specify the range
while True:
    user_input = input("Do you want to delete a range of images for every character? (yes/no): ").strip().lower()
    if user_input == 'yes':
        try:
            start_index = int(input("Enter the start index of the range: "))
            end_index = int(input("Enter the end index of the range: "))
            delete_images_in_range(start_index, end_index)
        except ValueError:
            print("Invalid input. Please enter valid integer indices.")
    elif user_input == 'no':
        print("No more images will be deleted.")
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")