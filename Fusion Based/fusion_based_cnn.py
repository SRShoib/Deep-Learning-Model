import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import cv2

# FusionCNN model imports
import torchvision.models as models

class FusionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FusionCNN, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet_features = mobilenet.features
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-2])
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2048 + 1280 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_out = self.resnet_features(x)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        
        mobilenet_out = self.mobilenet_features(x)
        mobilenet_out = nn.functional.adaptive_avg_pool2d(mobilenet_out, (1, 1))
        mobilenet_out = mobilenet_out.view(mobilenet_out.size(0), -1)
        
        efficientnet_out = self.efficientnet_features(x)
        efficientnet_out = nn.functional.adaptive_avg_pool2d(efficientnet_out, (1, 1))
        efficientnet_out = efficientnet_out.view(efficientnet_out.size(0), -1)
        
        fused_features = torch.cat([resnet_out, mobilenet_out, efficientnet_out], dim=1)
        
        return self.fc(fused_features)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = r"dataset path"  # <-- Set your dataset path
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

# Local save path for model
save_path = r"D:savemodel.pth"  # Local save path where model will save after tarining with name.pth

# Split dataset into train, validation, and test sets (70%, 15%, 15%)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = FusionCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loop
num_epochs = 50
best_val_acc = 0.0
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training phase
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

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = correct_train.double() / total_train

    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc.item())

    # Validation phase
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

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = correct_val.double() / total_val

    val_loss_history.append(epoch_val_loss)
    val_acc_history.append(epoch_val_acc.item())

    print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # Save the best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load the best model
model.load_state_dict(torch.load(save_path))
model.eval()

# Get predictions on the test set
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()

# Binarize the test labels
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

# Get softmax scores for the test set
y_scores = []

with torch.no_grad():
    for inputs, _ in test_loader:
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

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Grad-CAM Function for Any Feature Extractor
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
    target = output[0][target_class]
    target.backward()

    # Get gradients and activations
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    # Compute weights and Grad-CAM
    weights = np.mean(grads, axis=(2, 3))
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * acts[0, i, :, :]
    
    # Apply ReLU and normalize
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

# Grad-CAM++ Function for Any Feature Extractor
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
    target = output[0][target_class]
    target.backward()

    # Get gradients and activations
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

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

# Overlay Heatmap Function
def overlay_heatmap(heatmap, input_image):
    # Ensure the heatmap values are between 0 and 1
    heatmap = np.clip(heatmap, 0, 1)

    input_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * heatmap / 255 + 0.5 * input_image
    overlay = np.clip(overlay, 0, 1)
    return overlay

# Function to Generate and Plot Grad-CAM and Grad-CAM++ for Each Feature Extractor
def generate_and_plot_heatmaps(model, test_loader, class_names):
    model.eval()
    seen_classes = set()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for i in range(inputs.size(0)):
            label = labels[i].item()
            if label not in seen_classes:
                input_image = inputs[i:i+1]
                class_name = class_names[label]

                # Generate Grad-CAM and Grad-CAM++ heatmaps for each feature extractor
                grad_cam_resnet = grad_cam(input_image, model, target_class=label, feature_extractor=model.resnet_features[-1])
                grad_cam_plus_resnet = grad_cam_plus(input_image, model, target_class=label, feature_extractor=model.resnet_features[-1])

                grad_cam_mobilenet = grad_cam(input_image, model, target_class=label, feature_extractor=model.mobilenet_features[-1])
                grad_cam_plus_mobilenet = grad_cam_plus(input_image, model, target_class=label, feature_extractor=model.mobilenet_features[-1])

                grad_cam_efficientnet = grad_cam(input_image, model, target_class=label, feature_extractor=model.efficientnet_features[-1])
                grad_cam_plus_efficientnet = grad_cam_plus(input_image, model, target_class=label, feature_extractor=model.efficientnet_features[-1])

                # Overlay heatmaps
                grad_cam_resnet_overlay = overlay_heatmap(grad_cam_resnet, input_image)
                grad_cam_plus_resnet_overlay = overlay_heatmap(grad_cam_plus_resnet, input_image)

                grad_cam_mobilenet_overlay = overlay_heatmap(grad_cam_mobilenet, input_image)
                grad_cam_plus_mobilenet_overlay = overlay_heatmap(grad_cam_plus_mobilenet, input_image)

                grad_cam_efficientnet_overlay = overlay_heatmap(grad_cam_efficientnet, input_image)
                grad_cam_plus_efficientnet_overlay = overlay_heatmap(grad_cam_plus_efficientnet, input_image)

                # Original Image
                original_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

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

                plt.tight_layout()
                plt.show()

                seen_classes.add(label)

                if len(seen_classes) == len(class_names):
                    return

# Load the Best Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(save_path))
model.to(device)
model.eval()

# Get Class Names
class_names = full_dataset.classes

# Generate and Plot Heatmaps for Each Class
generate_and_plot_heatmaps(model, test_loader, class_names)
