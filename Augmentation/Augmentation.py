import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import math

# Define the augmentation techniques
augmentation_techniques = [
    transforms.RandomRotation(degrees=30),                      # Random rotation within ±30 degrees
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.CenterCrop(15),
    transforms.RandomVerticalFlip(p=0.5),                   
    # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),   # Random zooming with crop scale between 80% and 120%
    # transforms.ColorJitter(contrast=0.5),                       # Random contrast adjustment within ±50%
    transforms.GaussianBlur(kernel_size=15),                      # Apply Gaussian blur
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))   # Random affine transformation
]


# Function to save augmented images
def save_augmented_images(dataset_path, output_path, target_num_images_per_class):
    # Load dataset without any transformations to get original images
    dataset = ImageFolder(root=dataset_path)
    
    # Group images by class
    class_images = {}
    for img_path, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(img_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for class_name, img_paths in class_images.items():
        class_output_dir = os.path.join(output_path, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        print(f"Generating {target_num_images_per_class} images for class: {class_name}")

        # Count the number of original images
        num_original_images = len(img_paths)
        print(f"Found {num_original_images} original images for class {class_name}")

        # Save all original images first
        count = 0
        for img_path in img_paths:
            with Image.open(img_path).convert('RGB') as img:
                save_path = os.path.join(class_output_dir, f"orig_{count}.png")
                img.save(save_path)
                count += 1

        # Calculate how many augmentations are needed per image
        augmentations_needed_per_class = target_num_images_per_class - num_original_images
        augmentations_needed_per_image = math.floor(augmentations_needed_per_class / num_original_images)

        print(f"Augmentations needed per image: {augmentations_needed_per_image}")

        count_atmost_per_aug_tech = math.ceil(augmentations_needed_per_class/6)

        check = augmentations_needed_per_class - (augmentations_needed_per_image * num_original_images)
        print(f"Check pint: {check}")

        # Initialize `tech_conut` dictionary with all augmentation techniques
        tech_conut = {aug: 0 for aug in augmentation_techniques}

        # Apply augmentations to each original image
        for img_path in img_paths:
            with Image.open(img_path).convert('RGB') as img:  

                # if number check is greater than 0
                temp = augmentations_needed_per_image
                if check > 0:
                    temp += 1
                    check -= 1

                # Apply the selected number of random augmentations
                while(1):
                    flag = True
                    selected_augmentations = random.sample(augmentation_techniques, temp)

                    for aug in selected_augmentations:
                        if tech_conut[aug] -1 > count_atmost_per_aug_tech:
                            print(f"this: {aug} :technique is fullfill")
                            flag = False
                            break

                    if flag == False:
                        continue

                    for aug in selected_augmentations:
                            augmented_img = aug(img)
                            tech_conut[aug] += 1
                    
                            # # Unnormalize for saving
                            # unnorm_img = augmented_img.clone()
                            # unnorm_img = unnorm_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            # unnorm_img = unnorm_img.clamp(0, 1)
                            # unnorm_img = transforms.ToPILImage()(unnorm_img)

                            # Save the augmented image
                            save_path = os.path.join(class_output_dir, f"aug_{count}.png")
                            augmented_img.save(save_path)
                            count += 1

                    break

        print(f"{target_num_images_per_class} augmented images have been saved to {class_output_dir}, count = {count}")

# User inputs
dataset_path = r"Input path"  # Path to your dataset
output_path = r"Output path"    # Path to save augmented images
target_num_images_per_class = 3000  # Number of augmented images per class

# Run the function
save_augmented_images(dataset_path, output_path, target_num_images_per_class)
