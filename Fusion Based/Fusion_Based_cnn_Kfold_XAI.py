import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # For overlaying heatmap
import torch.nn.functional as F
import shap
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dataset_path = r"D:\Research\Propagranate\Pomegranate_Diseases_Dataset"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform) 
num_classes = len(full_dataset.classes)

# Hyperparameters
k_folds = 2
num_epochs = 50

batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
pooling_type = 'avg' # Options: "avg", "max", "sum"
weight_decay = 1e-4

model_save_path = r"D:\Research\Propagranate\model\Fusin_based\fusion_Raw_kfold2_updated.pth"


# Define the fusion model
class FusionCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate, pooling_type):
        super(FusionCNN, self).__init__()
        
        # Load pretrained models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.mobilenet_features = mobilenet.features
        self.efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-2])
        
        self.pooling_type = pooling_type
        
        # Adjusted feature dimensions
        feature_dim = 2048 + 1280 + 1280

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        resnet_out = self.resnet_features(x).view(x.size(0), -1)
        mobilenet_out = self.mobilenet_features(x)
        efficientnet_out = self.efficientnet_features(x)
        
        if self.pooling_type == 'avg':
            mobilenet_out = torch.mean(mobilenet_out, dim=[2, 3])
            efficientnet_out = torch.mean(efficientnet_out, dim=[2, 3])
        elif self.pooling_type == 'max':
            mobilenet_out = torch.amax(mobilenet_out, dim=[2, 3])
            efficientnet_out = torch.amax(efficientnet_out, dim=[2, 3])
        elif self.pooling_type == 'sum':
            mobilenet_out = torch.sum(mobilenet_out, dim=[2, 3])
            efficientnet_out = torch.sum(efficientnet_out, dim=[2, 3])
        
        fused_features = torch.cat([resnet_out, mobilenet_out, efficientnet_out], dim=1)
        return self.fc(fused_features)


kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Perform K-Fold Cross-Validation
all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
best_val_acc = 0.0
best_val_loader = None

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"\nFold {fold + 1}/{k_folds}")
    print("-" * 30)

    # Prepare Data Loaders
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize Model and Optimizer
    model = FusionCNN(num_classes, dropout_rate, pooling_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    

    # Training Loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = correct_train.double() / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct_val.double() / total_val

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc.item())
        val_accuracies.append(val_acc.item())

        # Save the best model for the current fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fold = fold
            best_val_loader = val_loader
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")

    # Log fold metrics
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

# Final output
print(f"\nBest Validation Accuracy Across Folds: {best_val_acc:.4f}")

# Plot training and validation accuracy for each fold
plt.figure(figsize=(10, 6))

for fold in range(k_folds):
    plt.plot(range(1, num_epochs + 1), all_train_accuracies[fold], label=f'Fold {fold + 1} - Training Acc')
    plt.plot(range(1, num_epochs + 1), all_val_accuracies[fold], label=f'Fold {fold + 1} - Validation Acc')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss for each fold
plt.figure(figsize=(10, 6))

for fold in range(k_folds):
    plt.plot(range(1, num_epochs + 1), all_train_losses[fold], label=f'Fold {fold + 1} - Training Loss')
    plt.plot(range(1, num_epochs + 1), all_val_losses[fold], label=f'Fold {fold + 1} - Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Across Folds')
plt.legend()
plt.grid(True)
plt.show()

# Load best model
model.load_state_dict(torch.load(model_save_path))
model.to(device)
model.eval()

# Evaluate on the best validation dataset
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in best_val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))

# Binarize the test labels
y_true_bin = label_binarize(y_true, classes=range(num_classes))

# Get softmax scores for the validation set
y_scores = []

with torch.no_grad():
    for inputs, _ in best_val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_scores.extend(outputs.softmax(dim=1).cpu().numpy())

y_scores = np.array(y_scores)

# Plot ROC Curves for each class
plt.figure(figsize=(10, 8))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{full_dataset.classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Grad-CAM Function
def grad_cam(input_image, model, target_class, feature_extractor):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Attach hooks to the specified feature extractor
    feature_extractor.register_forward_hook(forward_hook)
    feature_extractor.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_image)
    model.zero_grad()

    # Convert scalar tensor to Python number if necessary
    if output.ndim == 1:
        target = output[target_class]
    else:
        target = output[0][target_class]

    target.backward()

    # Process gradients and activations
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    # Compute Grad-CAM heatmap
    weights = np.mean(grads, axis=(2, 3))
    cam = np.zeros(acts.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights[0]):
        cam += w * acts[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))

    # Check for zero max value before normalizing
    if np.max(cam) != 0:
        cam /= np.max(cam)
    else:
        cam = np.zeros_like(cam)

    # Replace NaN or infinity values
    cam = np.nan_to_num(cam)
    
    return cam


# Grad-CAM++ Function
def grad_cam_plus(input_image, model, target_class, feature_extractor):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Attach hooks to the specified feature extractor
    feature_extractor.register_forward_hook(forward_hook)
    feature_extractor.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image)
    model.zero_grad()

    # Convert scalar tensor to Python number if necessary
    if output.ndim == 1:
        target = output[target_class]
    else:
        target = output[0][target_class]

    target.backward()

    # Process gradients and activations
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    # Grad-CAM++ calculation
    grads_power_2 = grads**2
    grads_power_3 = grads**3
    sum_acts = np.sum(acts, axis=(2, 3))
    eps = 1e-8

    alpha = grads_power_2 / (2 * grads_power_2 + sum_acts[:, :, np.newaxis, np.newaxis] * grads_power_3 + eps)
    weights = np.maximum(grads, 0) * alpha

    cam = np.sum(weights * acts, axis=1)[0]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    
    # Check for zero max value before normalizing
    if np.max(cam) != 0:
        cam /= np.max(cam)
    else:
        cam = np.zeros_like(cam)

    # Replace NaN or infinity values
    cam = np.nan_to_num(cam)

    return cam


# SHAP function
def shap_explanation(input_image, model, target_class):
    model.eval()

    # Convert input_image to numpy
    input_tensor = input_image.squeeze().cpu().numpy()  # (C, H, W)
    input_tensor = input_tensor.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    # Define SHAP Explainer
    def predict(images):
        tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(tensor)
        if outputs.ndim == 1:  # Handle missing batch dimension
            outputs = outputs.unsqueeze(0)
        return outputs[:, target_class].cpu().numpy()  # Return predictions for the target class only

    explainer = shap.Explainer(predict, masker=shap.maskers.Image("inpaint_telea", input_tensor.shape))
    shap_values = explainer(input_tensor[None, ...])  # Add batch dimension

    # Extract SHAP values for the target class
    shap_heatmap = shap_values[0].values[..., 0]  # SHAP for the selected class
    shap_heatmap = np.abs(shap_heatmap)  # Take absolute values
    shap_heatmap -= shap_heatmap.min()
    shap_heatmap /= shap_heatmap.max()
    shap_heatmap = cv2.resize(shap_heatmap, (224, 224))  # Resize to match the input image size

    return shap_heatmap



# LIME function
def lime_explanation(input_image, model, target_class):
    model.eval()

    # Convert input tensor to numpy image
    def predict(images):
        tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(tensor)
        return outputs.cpu().numpy()

    explainer = LimeImageExplainer()
    input_numpy = input_image.cpu().squeeze().permute(1, 2, 0).numpy()

    explanation = explainer.explain_instance(
        input_numpy,
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    print(f"Top labels: {explanation.top_labels}")  # Debugging line

    # Handle missing target_class in top_labels
    if target_class not in explanation.top_labels:
        print(f"Warning: target_class {target_class} not in top labels, using {explanation.top_labels[0]}")
        target_class = explanation.top_labels[0]

    # Get mask and superpixel overlay for the target class
    lime_result, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Convert lime_result to binary mask (0 or 1) based on a threshold
    lime_result = np.array(lime_result > 0.5, dtype=bool)  # Convert to boolean mask

    # Ensure lime_result has the same shape as the image (height, width)
    if lime_result.shape != input_numpy.shape[:2]:
        print(f"Warning: Reshaping lime_result from {lime_result.shape} to {input_numpy.shape[:2]}")
        lime_result = resize(lime_result, input_numpy.shape[:2], mode='constant', preserve_range=True)  # Resize to match

    # Ensure lime_result is 2D (remove any extra dimensions)
    lime_result = np.squeeze(lime_result)  # Remove extra dimensions if necessary

    return lime_result


# Overlay Heatmap Function
def overlay_heatmap(heatmap, input_image):
    input_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * heatmap / 255 + 0.5 * input_image
    overlay = np.clip(overlay, 0, 1)
    return overlay


# Function to generate and plot Grad-CAM and Grad-CAM++ for an image
def generate_and_plot_heatmaps(input_image, model, target_class, class_name):    
    grad_cam_resnet = grad_cam(input_image, model, target_class, feature_extractor=model.resnet_features[-1])
    grad_cam_plus_resnet = grad_cam_plus(input_image, model, target_class, feature_extractor=model.resnet_features[-1])

    grad_cam_mobilenet = grad_cam(input_image, model, target_class, feature_extractor=model.mobilenet_features[-1])
    grad_cam_plus_mobilenet = grad_cam_plus(input_image, model, target_class, feature_extractor=model.mobilenet_features[-1])

    grad_cam_efficientnet = grad_cam(input_image, model, target_class, feature_extractor=model.efficientnet_features[-1])
    grad_cam_plus_efficientnet = grad_cam_plus(input_image, model, target_class, feature_extractor=model.efficientnet_features[-1])

    # Overlay heatmaps
    grad_cam_resnet_overlay = overlay_heatmap(grad_cam_resnet, input_image)
    grad_cam_plus_resnet_overlay = overlay_heatmap(grad_cam_plus_resnet, input_image)

    grad_cam_mobilenet_overlay = overlay_heatmap(grad_cam_mobilenet, input_image)
    grad_cam_plus_mobilenet_overlay = overlay_heatmap(grad_cam_plus_mobilenet, input_image)

    grad_cam_efficientnet_overlay = overlay_heatmap(grad_cam_efficientnet, input_image)
    grad_cam_plus_efficientnet_overlay = overlay_heatmap(grad_cam_plus_efficientnet, input_image)

    shap_heatmap = shap_explanation(input_image, model, target_class)
    lime_result = lime_explanation(input_image, model, target_class)

    # Original Image
    original_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Ensure lime_result is 2D (height x width) before passing to mark_boundaries
    lime_result = np.squeeze(lime_result)  # Remove any extra dimensions if necessary

    # Check the shape of lime_result
    if lime_result.ndim == 3:  # If lime_result has 3 channels (like an RGB image), convert it to a binary mask
        lime_result = np.mean(lime_result, axis=-1) > 0.5  # Use average intensity to create a 2D binary mask

    # Plot Results
    plt.figure(figsize=(20, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(original_image)
    plt.title(f"{class_name} - Original")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(grad_cam_resnet_overlay)
    plt.title(f"{class_name} - Grad-CAM (ResNet50)")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(grad_cam_plus_resnet_overlay)
    plt.title(f"{class_name} - Grad-CAM++ (ResNet50)")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(grad_cam_mobilenet_overlay)
    plt.title(f"{class_name} - Grad-CAM (MobileNetV2)")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(grad_cam_plus_mobilenet_overlay)
    plt.title(f"{class_name} - Grad-CAM++ (MobileNetV2)")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(grad_cam_efficientnet_overlay)
    plt.title(f"{class_name} - Grad-CAM (EfficientNet)")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(grad_cam_plus_efficientnet_overlay)
    plt.title(f"{class_name} - Grad-CAM++ (EfficientNet)")
    plt.axis('off')

    # SHAP Heatmap
    plt.subplot(3, 3, 8)
    plt.imshow(shap_heatmap, cmap='jet')
    plt.axis('off')
    plt.title(f"{class_name} - SHAP")

    # LIME Overlay
    plt.subplot(3, 3, 9)
    # Apply mask to the original image and visualize boundaries correctly
    marked_image = mark_boundaries(original_image, lime_result, color=(1, 0, 0), mode='overlay')  # Mark boundaries with red
    plt.imshow(marked_image)
    plt.axis('off')
    plt.title(f"{class_name} - LIME")

    plt.tight_layout()
    plt.show()



# Select one image from each class
def select_images_from_each_class(dataset, num_classes):
    selected_images = []
    selected_labels = []
    seen_classes = set()

    for image, label in dataset:
        if label not in seen_classes:
            selected_images.append(image)
            selected_labels.append(label)
            seen_classes.add(label)
        if len(seen_classes) == num_classes:
            break

    return selected_images, selected_labels    


# Load the Best Model
model = FusionCNN(num_classes, dropout_rate, pooling_type).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Get images and labels
selected_images, selected_labels = select_images_from_each_class(full_dataset, num_classes)
class_names = full_dataset.classes

# Generate and plot heatmaps for each selected image
for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
    input_image = image.unsqueeze(0).to(device)
    class_name = class_names[label]
    print(f"Generating heatmaps for class: {class_name}")
    generate_and_plot_heatmaps(input_image, model, target_class=label, class_name=class_name)