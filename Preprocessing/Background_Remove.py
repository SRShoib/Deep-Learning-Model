from rembg import remove
from PIL import Image
import os
import io

input_folder = r'Input path'
output_folder = r'Output Path'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the image and read its data
        with open(input_path, 'rb') as input_image:
            image_data = input_image.read()

        # Remove the background
        output_data = remove(image_data)

        # Open the result image directly from output_data
        result_image = Image.open(io.BytesIO(output_data))  # Use BytesIO to read image data

        # Create a white background image
        white_bg = Image.new("RGBA", result_image.size, (255, 255, 255, 255))  # RGBA for transparency handling

        # Paste the result image on the white background (with transparency handling)
        if result_image.mode in ('RGBA', 'LA') or (result_image.mode == 'P' and 'transparency' in result_image.info):
            white_bg.paste(result_image, mask=result_image.split()[3])  # Use the alpha channel as a mask
        else:
            white_bg.paste(result_image)

        # Save the image with the white background
        white_bg.convert("RGB").save(output_path, format="JPEG")  # Convert to RGB for JPEG

        print(f"Processed {filename}")