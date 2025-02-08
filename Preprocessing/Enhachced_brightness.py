from PIL import Image, ImageEnhance
import os


brightness_factor = 1.1  # You can adjust the factor here

# Function to enhance brightness
def enhance_brightness(image_path, factor=brightness_factor):
    """
    Enhance the brightness of an image by a given factor.

    Parameters:
    - image_path: str, path to the image file.
    - factor: float, the factor by which to enhance the brightness. 1.0 means no change.

    Returns:
    - Enhanced image.
    """
    with Image.open(image_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

# Main dataset directory
dataset_dir = r'input path'

# Directory to save enhanced images
output_dir = r'Output path'

# Function to process dataset with class subfolders
def process_dataset(dataset_dir, output_dir, factor=brightness_factor):
    # Walk through all subfolders in the dataset
    for class_name in os.listdir(dataset_dir):
        class_folder = os.path.join(dataset_dir, class_name)

        if os.path.isdir(class_folder):  # Check if it's a folder (class subfolder)
            # Create corresponding class folder in the output directory
            output_class_folder = os.path.join(output_dir, class_name)
            os.makedirs(output_class_folder, exist_ok=True)
            
            # Process all images in the class folder
            for filename in os.listdir(class_folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                    image_path = os.path.join(class_folder, filename)
                    enhanced_image = enhance_brightness(image_path, factor)

                    # Save the enhanced image in the corresponding output class folder
                    output_path = os.path.join(output_class_folder, filename)
                    enhanced_image.save(output_path)

                    print(f"Enhanced image saved: {output_path}")

# Start processing the dataset
process_dataset(dataset_dir, output_dir, factor=brightness_factor) 
