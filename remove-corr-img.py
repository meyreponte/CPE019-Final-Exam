import os
from PIL import Image

def find_and_remove_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify if it's a valid image
                except (IOError, SyntaxError) as e:
                    print(f"Removing corrupt image: {file_path}")
                    os.remove(file_path)

# Check and remove corrupt images in train, validation, and test directories
find_and_remove_corrupt_images('output/train')
find_and_remove_corrupt_images('output/val')
find_and_remove_corrupt_images('output/test')
