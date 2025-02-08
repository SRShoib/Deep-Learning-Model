from rembg import remove
from PIL import Image
import os
import io

# Define the base input and output directories
input_base_folder = r'Input path'
output_base_folder = r'Output path'

# Ensure the base output directory exists
os.makedirs(output_base_folder, exist_ok=True)

# Define the target size for resizing (width, height)
target_size = (512, 512)

# Loop through each subclass folder in the dataset
for subclass in os.listdir(input_base_folder):
    input_folder = os.path.join(input_base_folder, subclass)
    output_folder = os.path.join(output_base_folder, subclass)
    
    # Ensure it's a directory
    if not os.path.isdir(input_folder):
        continue
    
    # Ensure the subclass output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image file
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image and read its data
            with open(input_path, 'rb') as input_image:
                image_data = input_image.read()

            # Remove the background
            output_data = remove(image_data)

            # Open the result image directly from output_data
            result_image = Image.open(io.BytesIO(output_data))

            # Create a white background image
            white_bg = Image.new("RGBA", result_image.size, (255, 255, 255, 255))

            # Paste the result image on the white background (handling transparency)
            if result_image.mode in ('RGBA', 'LA') or (result_image.mode == 'P' and 'transparency' in result_image.info):
                white_bg.paste(result_image, mask=result_image.split()[3])
            else:
                white_bg.paste(result_image)

            # Resize the image to target size
            white_bg = white_bg.resize(target_size, Image.ANTIALIAS)

            # Save the image with the white background
            white_bg.convert("RGB").save(output_path, format="JPEG")

            print(f"Processed {filename} in {subclass}")
