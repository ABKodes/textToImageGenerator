from PIL import Image, ImageDraw, ImageFont
import json

import os

class DatasetGenerator:
    def __init__(self, text_file, font_dir, dataset_dir='amharic_dataset'):
        self.text_file = text_file
        self.font_dir = font_dir
        self.dataset_dir = dataset_dir
        self.processed_fonts_file = 'processed_fonts.txt'
        self.restricted_char_map = {
            '"': 'QUOTE',
            '<': 'LT',
            '>': 'GT',
            '?': 'QUESTION',
            '*': 'STAR',
            '/': 'FORWARD_SLASH',
            '.': 'DOT'
        }
        self.processed_fonts = self._load_processed_fonts()
        self.dataset_labels = []

    def _load_processed_fonts(self):
        if os.path.exists(self.processed_fonts_file):
            with open(self.processed_fonts_file, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        return set()

    def _sanitize_char_for_filename(self, char):
        """Replace restricted characters with safe substitutes for filenames."""
        return self.restricted_char_map.get(char, char)

    def generate_images(self):
        with open(self.text_file, 'r', encoding='utf-8') as file:
            text = file.read()

        font_files = [f for f in os.listdir(self.font_dir) if f.endswith(('.ttf', '.otf', '.TTF'))]
        for font_file in font_files:
            if font_file in self.processed_fonts:
                print(f"Skipping already processed font: {font_file}")
                continue

            font_path = os.path.join(self.font_dir, font_file)
            print(f"Processing new font: {font_file}")

            font_size = 20
            font = ImageFont.truetype(font_path, font_size)

            for char in text:
                if not char.isspace():
                    self._process_char_for_font(char, font, font_file)

            self.processed_fonts.add(font_file)
            with open(self.processed_fonts_file, 'a', encoding='utf-8') as file:
                file.write(f"{font_file}\n")

        self._save_dataset_labels()

    def _process_char_for_font(self, char, font, font_file):
        safe_char = self._sanitize_char_for_filename(char)
        char_folder_name = safe_char
        char_folder_path = os.path.join(self.dataset_dir, char_folder_name)
        os.makedirs(char_folder_path, exist_ok=True)

        existing_files = os.listdir(char_folder_path)
        max_existing_index = max([int(file.split('_')[-1].split('.')[0]) for file in existing_files if file.endswith('.png')], default=0)
        start_index = max_existing_index + 1

        for i in range(start_index, start_index + 20 ):
            char_image_name = f"{safe_char}_{i}.png"
            char_image_path = os.path.join(char_folder_path, char_image_name)
            image = Image.new('L', (32, 32), color='white')
            draw = ImageDraw.Draw(image)
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (32 - text_width) // 2
            text_y = (32 - text_height) // 2
            draw.text((text_x, text_y), char, fill='black', font=font)
            image.save(char_image_path)
            print(f"Saved image for '{char}' as {char_image_path}")
            self.dataset_labels.append({"image_path": char_image_path, "font_type": font_file, "label": char})

    def _save_dataset_labels(self):
        sorted_labels = sorted(self.dataset_labels, key=lambda x: x['label'])
        output_json_file = 'amharic_dataset_labels.json'
        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(sorted_labels, json_file, ensure_ascii=False, indent=4)
        print(f"Labels have been sorted and saved to {output_json_file}")

    def delete(self, start_index=None, end_index=None):
        """Delete images within the specified range for every character."""
        if start_index is None or end_index is None:
            print("Both start and end indices must be specified for deletion.")
            return

        print(f"Deleting images from index {start_index} to {end_index} for every character...")

        # Iterate over all character folders in the dataset directory
        for char_folder_name in os.listdir(self.dataset_dir):
            char_folder_path = os.path.join(self.dataset_dir, char_folder_name)

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

# Usage Example:
generator = DatasetGenerator(text_file='text.txt', font_dir='fonts')
generator.generate_images()

# Ask user if they want to delete images and specify the range
while True:
    user_input = input("Do you want to delete a range of images for every character? (yes/no): ").strip().lower()
    if user_input == 'yes':
        try:
            start_index = int(input("Enter the start index of the range: "))
            end_index = int(input("Enter the end index of the range: "))
            generator.delete(start_index, end_index)
        except ValueError:
            print("Invalid input. Please enter valid integer indices.")
    elif user_input == 'no':
        print("No more images will be deleted.")
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
