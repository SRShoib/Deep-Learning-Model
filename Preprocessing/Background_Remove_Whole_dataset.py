from rembg import remove
from PIL import Image
import os
import io

# Root folder containing subfolders for each class
dataset_folder = r'input path'

# Output folder for the preprocessed data
output_folder = r'Output path'

# Loop through each class (subfolder) in the dataset folder
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)

    # Skip if it's not a directory (in case of unexpected files)
    if not os.path.isdir(class_path):
        continue

    # Create the corresponding output folder for the class
    class_output_folder = os.path.join(output_folder, class_folder)
    os.makedirs(class_output_folder, exist_ok=True)

    # Loop through the images in each class folder
    for filename in os.listdir(class_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            input_path = os.path.join(class_path, filename)
            output_path = os.path.join(class_output_folder, filename)

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

            print(f"Processed {filename} in {class_folder}")
