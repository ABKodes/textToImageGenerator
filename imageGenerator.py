from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import json
import os
import random
import numpy as np

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

        for i in range(start_index, start_index + 1):
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

            # Apply transformations to make the image more realistic
            image = self._apply_transformations(image)

            image.save(char_image_path)
            print(f"Saved image for '{char}' as {char_image_path}")
            self.dataset_labels.append({"image_path": char_image_path, "font_type": font_file, "label": char})

    def _apply_transformations(self, image):
        """Apply mild transformations to the image to simulate real-world conditions."""
        # Randomly apply Gaussian noise
        # if random.choice([True, False]):
        #     image = self._add_gaussian_noise(image)

        # Randomly apply salt and pepper noise
        # if random.choice([True, False]):
        #     image = self._add_salt_pepper_noise(image)

        # Randomly apply Gaussian blur
        if random.choice([True, False]):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        # Randomly rotate the image
        if random.choice([True, False]):
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, expand=True, fillcolor='white')

        # Randomly adjust brightness
        if random.choice([True, False]):
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly adjust contrast
        if random.choice([True, False]):
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly add occlusions
        # if random.choice([True, False]):
        #     image = self._add_random_occlusions(image, num_occlusions=1, max_occlusion_size=5)

        # # Randomly simulate JPEG compression
        # if random.choice([True, False]):
        #     image = self._simulate_jpeg_compression(image)

        return image


    def _add_gaussian_noise(self, image):
        """Add mild Gaussian noise to the image."""
        np_image = np.array(image)
        noise = np.random.normal(0, 5, np_image.shape).astype(np.uint8)  # Reduced standard deviation
        noisy_image = np.clip(np_image + noise, 0, 255)
        return Image.fromarray(noisy_image)

    def _add_salt_pepper_noise(self, image, amount=0.01):  # Reduced amount of noise
        """Add mild salt and pepper noise to the image."""
        np_image = np.array(image)
        s_vs_p = 0.5
        out = np.copy(np_image)
        # Salt mode
        num_salt = np.ceil(amount * np_image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * np_image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_image.shape]
        out[coords] = 0
        return Image.fromarray(out)    

    def _add_salt_pepper_noise(self, image, amount=0.02):
        """Add salt and pepper noise to the image."""
        np_image = np.array(image)
        s_vs_p = 0.5
        out = np.copy(np_image)
        # Salt mode
        num_salt = np.ceil(amount * np_image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * np_image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_image.shape]
        out[coords] = 0
        return Image.fromarray(out)

    def _add_random_occlusions(self, image, num_occlusions=1, max_occlusion_size=5):
        """Add small, sparse occlusions to the image."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for _ in range(num_occlusions):
            # Randomly choose the size of the occlusion
            occlusion_width = random.randint(1, max_occlusion_size)
            occlusion_height = random.randint(1, max_occlusion_size)
            # Randomly choose the position of the occlusion
            x1 = random.randint(0, width - occlusion_width)
            y1 = random.randint(0, height - occlusion_height)
            x2 = x1 + occlusion_width
            y2 = y1 + occlusion_height
            draw.rectangle([x1, y1, x2, y2], fill="black")
        return image


    def _simulate_jpeg_compression(self, image, quality=random.randint(50, 90)):
        """Simulate JPEG compression artifacts."""
        temp_path = "temp.jpg"

        # Save the image to the temporary file
        image.save(temp_path, "JPEG", quality=quality)

        # Open and copy the image to release the file handle
        with Image.open(temp_path) as compressed_image:
            compressed_copy = compressed_image.copy()  # Copy the image before closing

        # Now it's safe to remove the file
        os.remove(temp_path)

        return compressed_copy


    def _save_dataset_labels(self):
        sorted_labels = sorted(self.dataset_labels, key=lambda x: x['label'])
        output_json_file = 'amharic_dataset_labels.json'
        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(sorted_labels, json_file, ensure_ascii=False, indent=4)
        print(f"Labels have been sorted and saved to {output_json_file}")


# Example usage
if __name__ == "__main__":
    text_file = "text.txt"  # Path to your Amharic text file
    font_dir = "fonts"  # Directory containing your Amharic fonts
    generator = DatasetGenerator(text_file, font_dir)
    generator.generate_images()