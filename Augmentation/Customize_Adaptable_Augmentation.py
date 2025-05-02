import os
import cv2
import numpy as np
import random
import math
from tqdm import tqdm
from collections import defaultdict

# Define paths
input_dataset_folder = r"D:\Research\Topics\Black_Gram_Dataset\Mendeley_Data\Brighness Enhanced"  # Root folder with class subfolders
output_dataset_folder = r"D:\Research\Topics\Black_Gram_Dataset\Mendeley_Data\augmented"  # Where augmented dataset will be saved
os.makedirs(output_dataset_folder, exist_ok=True)

# Set total target number of augmented images per class
target_images_per_class = 3000  # Change this to your desired number per class

# Augmentation functions
def random_rotation(image):
    angle = random.uniform(-25, 25)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

def random_contrast(image):
    contrast = random.uniform(0.8, 1.2)
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

def random_zoom(image):
    zoom_factor = random.uniform(1.1, 1.5)
    height, width = image.shape[:2]
    new_height, new_width = int(height*zoom_factor), int(width*zoom_factor)
    resized = cv2.resize(image, (new_width, new_height))
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    return resized[start_y:start_y+height, start_x:start_x+width]

# List of augmentation functions with simplified names
augmentation_functions = [
    ("rot", random_rotation),
    ("hflip", horizontal_flip),
    ("vflip", vertical_flip),
    ("zoom", random_zoom),
    ("contr", random_contrast)
]

# Get list of class folders
class_folders = [f for f in os.listdir(input_dataset_folder) 
                if os.path.isdir(os.path.join(input_dataset_folder, f))]

if not class_folders:
    print("No class folders found in the input dataset folder.")
    exit()


# Process each class folder
for class_folder in tqdm(class_folders, desc="Processing classes"):
    input_class_folder = os.path.join(input_dataset_folder, class_folder)
    output_class_folder = os.path.join(output_dataset_folder, class_folder)
    os.makedirs(output_class_folder, exist_ok=True)
    
    original_images = [f for f in os.listdir(input_class_folder) 
                     if os.path.isfile(os.path.join(input_class_folder, f))]
    num_original_images = len(original_images)
    
    if num_original_images == 0:
        print(f"No images found in class folder: {class_folder}")
        continue
    
    # Step 1: Copy original images
    for filename in tqdm(original_images, desc=f"Copying originals for {class_folder}", leave=False):
        img_path = os.path.join(input_class_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        cv2.imwrite(os.path.join(output_class_folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    augmentations_needed = target_images_per_class - num_original_images
    if augmentations_needed <= 0:
        print(f"Class {class_folder} already has enough images ({num_original_images})")
        continue
    
    # Calculate base and extra augmentations
    base_aug_per_image = max(1, augmentations_needed // num_original_images)
    remaining_augmentations = augmentations_needed % num_original_images
    
    # NEW: Balanced technique distribution
    technique_pool = [aug[0] for aug in augmentation_functions]
    total_techniques = len(technique_pool)
    
    # Assign techniques to images
    technique_assignments = defaultdict(list)
    
    # Phase 1: Assign at least one technique to each image
    for i, filename in enumerate(original_images):
        technique = technique_pool[i % total_techniques]
        technique_assignments[filename].append(technique)
    
    # Phase 2: Distribute remaining base augmentations
    for filename in original_images:
        while len(technique_assignments[filename]) < base_aug_per_image:
            # Find least used technique not already assigned
            tech_counts = defaultdict(int)
            for tech_list in technique_assignments.values():
                for tech in tech_list:
                    tech_counts[tech] += 1
            
            # Get available techniques for this image
            available_techs = [t for t in technique_pool 
                             if t not in technique_assignments[filename]]
            
            if available_techs:
                # Use least used available technique
                tech = min(available_techs, key=lambda x: tech_counts[x])
            else:
                # All techniques used, pick least used overall
                tech = min(tech_counts, key=tech_counts.get)
            
            technique_assignments[filename].append(tech)
    
    # Phase 3: Distribute any remaining augmentations
    for i in range(remaining_augmentations):
        filename = original_images[i % num_original_images]
        
        # Find least used technique not already assigned
        tech_counts = defaultdict(int)
        for tech_list in technique_assignments.values():
            for tech in tech_list:
                tech_counts[tech] += 1
        
        available_techs = [t for t in technique_pool 
                         if t not in technique_assignments[filename]]
        
        if available_techs:
            tech = min(available_techs, key=lambda x: tech_counts[x])
        else:
            tech = min(tech_counts, key=tech_counts.get)
        
        technique_assignments[filename].append(tech)
    
    # Apply the augmentations
    technique_counter = defaultdict(int)
    augmented_count = num_original_images
    
    for filename, techniques in tqdm(technique_assignments.items(), desc=f"Augmenting {class_folder}"):
        img_path = os.path.join(input_class_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        base_name, ext = os.path.splitext(filename)
        
        for tech in techniques:
            aug_func = next((func for name, func in augmentation_functions if name == tech), None)
            if not aug_func:
                continue
            
            augmented_img = aug_func(img)
            output_filename = f"{base_name}_{tech}{ext}"
            output_path = os.path.join(output_class_folder, output_filename)
            
            # Handle duplicates
            counter = 1
            while os.path.exists(output_path):
                output_filename = f"{base_name}_{tech}_{counter}{ext}"
                output_path = os.path.join(output_class_folder, output_filename)
                counter += 1
            
            cv2.imwrite(output_path, augmented_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            augmented_count += 1
            technique_counter[tech] += 1
    
    # Print statistics
    print(f"\nClass {class_folder} augmentation summary:")
    print(f"- Original images: {num_original_images}")
    print(f"- Augmented images: {augmented_count - num_original_images}")
    print(f"- Total images: {augmented_count}")
    print("Technique distribution:")
    for tech in technique_pool:
        count = technique_counter.get(tech, 0)
        print(f"  {tech}: {count} ({(count/augmentations_needed)*100:.1f}%)")

print("\nAugmentation completed for all classes.")