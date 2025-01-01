# filepath: /path/to/image_to_text.py
import cv2
import pytesseract

# Path to the Tesseract executable (optional if added to PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load an image from file
image = cv2.imread('path_to_your_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Tesseract to extract text
text = pytesseract.image_to_string(gray)

print(text)