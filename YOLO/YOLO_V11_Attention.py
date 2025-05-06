import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange
import cv2 
import shap
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

# Set seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dataset_path = r"D:\Research\Topics\Fruits Original"  # Path to your dataset

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

# Parameters
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
weight_decay = 1e-4
num_epochs = 50
validation_split = 0.2  # 20% of the data will be used for validation

# Model save path
model_save_path = r"D:\Research\Topics\Propagranate\model\yolo\Yolo11model.pth"

# Split dataset into training and validation sets
train_size = int((1 - validation_split) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(avg_out + max_out) * x

        # Spatial Attention
        avg_out = torch.mean(channel_att, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_att, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * channel_att

        return spatial_att

# Transformer Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return self.proj(out)

# CSP Bottleneck with CBAM
class CSPBottleneckWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBottleneckWithCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(out_channels)  # Add CBAM here

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)  # Apply CBAM
        return x

# YOLOv11 Backbone with Attention
class YOLOv11Backbone(nn.Module):
    def __init__(self, num_classes=3):
        super(YOLOv11Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.csp1 = CSPBottleneckWithCBAM(32, 64)
        self.csp2 = CSPBottleneckWithCBAM(64, 128)
        self.attention = SelfAttention(128)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        x = rearrange(x, 'b c -> b 1 c')
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# Initialize Model and Optimizer
model = YOLOv11Backbone(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Training and Validation Loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_acc = 0.0

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

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved at {model_save_path}")

# Plot training and validation accuracy
plt.figure(figsize=(10, 8))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 8))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load the best model
model = YOLOv11Backbone(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

# Evaluate on the validation dataset
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
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
    for inputs, _ in val_loader:
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


def grad_cam(input_image, model, target_class):
    model.eval()
    gradients = []
    activations = []

    # Register hooks to capture gradients and activations
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Replace with the correct layer in your model
    target_layer = model.csp2.conv2  # Target layer for Grad-CAM
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image)
    model.zero_grad()

    # Get the output for the target class
    if output.ndim == 1:
        target = output[target_class]
    else:
        target = output[0][target_class]

    # Backward pass
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
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def grad_cam_plus(input_image, model, target_class):
    model.eval()
    gradients = []
    activations = []

    # Register hooks to capture gradients and activations
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Replace with the correct layer in your model
    target_layer = model.csp2.conv2  # Target layer for Grad-CAM++
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_image)
    model.zero_grad()

    # Get the output for the target class
    if output.ndim == 1:
        target = output[target_class]
    else:
        target = output[0][target_class]

    # Backward pass
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
    cam -= np.min(cam)
    cam /= np.max(cam)
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
    grad_cam_heatmap = grad_cam(input_image, model, target_class)
    grad_cam_plus_heatmap = grad_cam_plus(input_image, model, target_class)
    shap_heatmap = shap_explanation(input_image, model, target_class)
    lime_result = lime_explanation(input_image, model, target_class)

    original_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Ensure lime_result is 2D (height x width) before passing to mark_boundaries
    lime_result = np.squeeze(lime_result)  # Remove any extra dimensions if necessary

    # Check the shape of lime_result
    if lime_result.ndim == 3:  # If lime_result has 3 channels (like an RGB image), convert it to a binary mask
        lime_result = np.mean(lime_result, axis=-1) > 0.5  # Use average intensity to create a 2D binary mask

    # Convert the original image to RGB (3 channels) if it's grayscale
    if original_image.ndim == 2:
        original_image = np.repeat(original_image[:, :, np.newaxis], 3, axis=-1)  # Convert grayscale to RGB

    plt.figure(figsize=(20, 10))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(f"{class_name} - Original")

    # Grad-CAM Overlay
    grad_cam_overlay = overlay_heatmap(grad_cam_heatmap, input_image)
    plt.subplot(2, 3, 2)
    plt.imshow(grad_cam_overlay)
    plt.axis('off')
    plt.title(f"{class_name} - Grad-CAM")

    # Grad-CAM++ Overlay
    grad_cam_plus_overlay = overlay_heatmap(grad_cam_plus_heatmap, input_image)
    plt.subplot(2, 3, 3)
    plt.imshow(grad_cam_plus_overlay)
    plt.axis('off')
    plt.title(f"{class_name} - Grad-CAM++")

    # SHAP Heatmap
    plt.subplot(2, 3, 4)
    plt.imshow(shap_heatmap, cmap='jet')
    plt.axis('off')
    plt.title(f"{class_name} - SHAP")

    # LIME Overlay
    plt.subplot(2, 3, 5)
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

# Load the best model
model = YOLOv11Backbone(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

# Extract a batch of images and labels from the validation loader
val_images, val_labels = next(iter(val_loader))  # Use the best validation loader
val_images = val_images.to(device)
val_labels = val_labels.to(device)

# Generate and plot heatmaps for the first image in the batch
input_image = val_images[0].unsqueeze(0)  # Add batch dimension
target_class = val_labels[0].item()  # Get the target class
class_name = full_dataset.classes[target_class]  # Get the class name
print(f"Generating heatmaps for class: {class_name}")
generate_and_plot_heatmaps(input_image, model, target_class=target_class, class_name=class_name)