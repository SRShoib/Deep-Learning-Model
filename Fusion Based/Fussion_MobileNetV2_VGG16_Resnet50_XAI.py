import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import cv2
import torch.nn.functional as F
import shap
from lime.lime_image import LimeImageExplainer
from captum.attr import IntegratedGradients
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset_path = r"D:\Research\Topics\Black_Gram_Dataset\Mendeley_Data\Brighness Enhanced"
model_save_path = r"D:\Research\Topics\Black_Gram_Dataset\Mendeley_Data\Model Training\Model\fusion_model.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

# Hyperparameters
num_epochs = 50
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
pooling_type = 'avg'  # Options: "avg", "max", "sum"
weight_decay = 1e-4


class FusionModel(nn.Module):
    def __init__(self, num_classes, dropout_rate, pooling_type):
        super(FusionModel, self).__init__()
        
        # Load pre-trained models
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the classification heads
        self.mobilenet.classifier = nn.Identity()
        self.vgg16.classifier = nn.Identity()
        self.resnet50.fc = nn.Identity()
        
        # Freeze the base models
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.resnet50.parameters():
            param.requires_grad = False
            
        # Configure pooling layer based on parameter
        if pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling_type == 'sum':
            # Custom sum pooling implementation
            self.pool = lambda x: torch.sum(x, dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose 'avg', 'max', or 'sum'")
        
        # Calculate the actual feature dimensions
        with torch.no_grad():
            # Move models to CPU temporarily for feature size calculation
            mobilenet_cpu = self.mobilenet.to('cpu')
            vgg16_cpu = self.vgg16.to('cpu')
            resnet50_cpu = self.resnet50.to('cpu')
            
            # Create a dummy input on CPU
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Get features from each model
            mobilenet_features = mobilenet_cpu.features(dummy_input)
            mobilenet_features = self.pool(mobilenet_features).view(1, -1)
            
            vgg_features = vgg16_cpu.features(dummy_input)
            vgg_features = self.pool(vgg_features).view(1, -1)
            
            resnet_features = resnet50_cpu(dummy_input)
            if len(resnet_features.shape) > 2:  # If features are not already flattened
                resnet_features = self.pool(resnet_features).view(1, -1)
            
            # Calculate combined feature size
            combined_features = mobilenet_features.size(1) + vgg_features.size(1) + resnet_features.size(1)
            
            # Move models back to original device (will be set properly in .to(device) later)
            self.mobilenet = mobilenet_cpu
            self.vgg16 = vgg16_cpu
            self.resnet50 = resnet50_cpu
        
        # New classification head with configurable dropout
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Get features from each model
        # MobileNetV2 features
        mobilenet_features = self.mobilenet.features(x)
        mobilenet_features = self.pool(mobilenet_features).view(x.size(0), -1)
        
        # VGG16 features
        vgg_features = self.vgg16.features(x)
        vgg_features = self.pool(vgg_features).view(x.size(0), -1)
        
        # ResNet50 features
        resnet_features = self.resnet50(x)
        if len(resnet_features.shape) > 2:  # If features are not already flattened
            resnet_features = self.pool(resnet_features).view(x.size(0), -1)
        
        # Concatenate all features
        combined = torch.cat((mobilenet_features, vgg_features, resnet_features), dim=1)
        
        # Classification head
        out = self.classifier(combined)
        
        return out

# Split dataset into train, validation, and test sets (70%, 15%, 15%)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer with configurable learning rate and weight decay
model = FusionModel(num_classes, dropout_rate, pooling_type).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training and validation loop
best_val_acc = 0.0
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Current hyperparameters: LR={learning_rate}, BS={batch_size}, Dropout={dropout_rate}, Pooling={pooling_type}, WD={weight_decay}")
    
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
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

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
model = FusionModel(num_classes, dropout_rate, pooling_type).to(device)
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

# XAI Functions
def grad_cam(input_tensor, model, target_class, target_layer, backbone='resnet'):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Ensure input tensor requires gradients
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    model.eval()
    output = model(input_tensor)
    if isinstance(output, tuple):  # in case model returns multiple outputs
        output = output[0]

    # Extract class score
    class_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    class_score.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Check if hooks captured data
    if len(activations) == 0 or len(gradients) == 0:
        raise ValueError(f"Hooks failed to capture data for backbone '{backbone}'. Check target_layer.")

    # Process data
    act = activations[0].detach()
    grad = gradients[0].detach()

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy()

def grad_cam_plus_plus(input_tensor, model, target_class, target_layer, backbone='resnet'):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Ensure input_tensor has requires_grad=True
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    model.eval()
    output = model(input_tensor)
    if isinstance(output, tuple):  # support for multiple outputs
        output = output[0]
    class_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    class_score.backward(retain_graph=True)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Check if hooks captured data
    if len(activations) == 0 or len(gradients) == 0:
        raise ValueError(f"Hooks failed to capture data for backbone '{backbone}' in Grad-CAM++.")

    # Get hooked outputs
    act = activations[0].detach()          # shape: [B, C, H, W]
    grad = gradients[0].detach()           # shape: [B, C, H, W]

    # Compute Grad-CAM++
    grad_square = grad ** 2
    grad_cube = grad ** 3
    sum_grad = torch.sum(act * grad_cube, dim=(2, 3), keepdim=True) + 1e-8

    alpha = grad_square / (2 * grad_square + sum_grad)
    alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

    weights = torch.sum(alpha * F.relu(grad), dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam.cpu().numpy()



def shap_explanation(input_image, model, target_class):
    model.eval()
    input_tensor = input_image.squeeze().cpu().numpy()
    input_tensor = input_tensor.transpose(1, 2, 0)

    def predict(images):
        tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(tensor)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        return outputs[:, target_class].cpu().numpy()

    explainer = shap.Explainer(predict, masker=shap.maskers.Image("inpaint_telea", input_tensor.shape))
    shap_values = explainer(input_tensor[None, ...])
    
    shap_heatmap = shap_values[0].values[..., 0]
    shap_heatmap = np.abs(shap_heatmap)
    shap_heatmap -= shap_heatmap.min()
    shap_heatmap /= shap_heatmap.max()
    shap_heatmap = cv2.resize(shap_heatmap, (224, 224))

    return shap_heatmap

# LIME function (remains the same as it's model-agnostic)
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

def integrated_gradients(input_image, model, target_class, steps=50, internal_batch_size=10):
    """
    Compute Integrated Gradients attribution for an input image.
    Args:
        input_image: Input tensor (1, C, H, W)
        model: PyTorch model
        target_class: Class index to explain
        steps: Number of integration steps (default: 50)
        internal_batch_size: Batch size for intermediate gradient computation (default: 10)
    Returns:
        2D numpy array of attribution scores (H, W)
    """
    model.eval()
    model.zero_grad()
    torch.cuda.empty_cache()  # Optional but recommended before attribution

    ig = IntegratedGradients(model)

    # Compute attributions with internal batching
    attributions = ig.attribute(
        input_image,
        target=target_class,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=False
    )

    # Convert to heatmap
    ig_heatmap = attributions.squeeze().cpu().detach().numpy()
    if ig_heatmap.ndim == 3:  # For RGB images, take mean across channels
        ig_heatmap = np.mean(ig_heatmap, axis=0)

    # Normalize
    ig_heatmap = np.abs(ig_heatmap)
    ig_heatmap = cv2.resize(ig_heatmap, (224, 224))
    ig_heatmap = (ig_heatmap - ig_heatmap.min()) / (ig_heatmap.max() - ig_heatmap.min() + 1e-10)

    return ig_heatmap


def overlay_heatmap(heatmap, input_image):
    # Convert input image tensor to numpy
    input_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    # Resize heatmap to match input image size
    heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))  # (W, H)

    # Apply colormap to resized heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]

    # Blend images
    overlay = 0.5 * heatmap_color + 0.5 * input_image
    overlay = np.clip(overlay, 0, 1)

    return overlay


def generate_and_plot_heatmaps(input_image, model, target_class, class_name):    
    # # Get feature extractors from each model
    # resnet_feature = model.resnet50.layer4[-1]
    # mobilenet_feature = model.mobilenet.features[-1]
    # vgg_feature = model.vgg16.features[-1]

    # Generate explanations

    # Example call with target layers
    # Suppose you already loaded model and input_image correctly

    if isinstance(input_image, torch.Tensor):
        input_tensor = input_image
    else:
        # If input_image is a PIL image or ndarray, apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(input_image).unsqueeze(0)  # [1, 3, 224, 224]

    # Ensure tensor shape is correct
    if input_tensor.dim() == 5 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)


    # Generate Grad-CAM heatmap for ResNet
    resnet_layer = model.resnet50.layer4[-1]
    mobilenet_layer = model.mobilenet.features[-1]
    vgg_layer = model.vgg16.features[-1]

    grad_cam_resnet = grad_cam(input_tensor, model, target_class=label, target_layer=resnet_layer, backbone='resnet')
    grad_cam_mobilenet = grad_cam(input_tensor, model, target_class=label, target_layer=mobilenet_layer, backbone='mobilenet')
    grad_cam_vgg = grad_cam(input_tensor, model, target_class=label, target_layer=vgg_layer, backbone='vgg')


    
    grad_cam_plus_resnet = grad_cam_plus_plus(input_tensor, model, target_class=label, target_layer=resnet_layer, backbone='resnet')
    
    grad_cam_plus_mobilenet = grad_cam_plus_plus(input_tensor, model, target_class=label, target_layer=mobilenet_layer, backbone='mobilenet')
    
    grad_cam_plus_vgg = grad_cam_plus_plus(input_tensor, model, target_class=label, target_layer=vgg_layer, backbone='vgg')



    shap_heatmap = shap_explanation(input_image, model, target_class)
    lime_result = lime_explanation(input_image, model, target_class)
    ig_heatmap = integrated_gradients(input_image, model, target_class)

    # Create overlays
    original_image = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
    # Ensure lime_result is 2D (height x width) before passing to mark_boundaries
    lime_result = np.squeeze(lime_result)  # Remove any extra dimensions if necessary

    # Check the shape of lime_result
    if lime_result.ndim == 3:  # If lime_result has 3 channels (like an RGB image), convert it to a binary mask
        lime_result = np.mean(lime_result, axis=-1) > 0.5  # Use average intensity to create a 2D binary mask

    # # Convert the original image to RGB (3 channels) if it's grayscale
    # if original_image.ndim == 2:
    #     original_image = np.repeat(original_image[:, :, np.newaxis], 3, axis=-1)  # Convert grayscale to RGB


    # Plot results
    plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(3, 4, 1)
    plt.imshow(original_image)
    plt.title(f"{class_name} - Original")
    plt.axis('off')

    # ResNet explanations
    plt.subplot(3, 4, 2)
    plt.imshow(overlay_heatmap(grad_cam_resnet, input_image))
    plt.title("ResNet Grad-CAM")
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(overlay_heatmap(grad_cam_plus_resnet, input_image))
    plt.title("ResNet Grad-CAM++")
    plt.axis('off')

    # MobileNet explanations
    plt.subplot(3, 4, 4)
    plt.imshow(overlay_heatmap(grad_cam_mobilenet, input_image))
    plt.title("MobileNet Grad-CAM")
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(overlay_heatmap(grad_cam_plus_mobilenet, input_image))
    plt.title("MobileNet Grad-CAM++")
    plt.axis('off')

    # VGG explanations
    plt.subplot(3, 4, 6)
    plt.imshow(overlay_heatmap(grad_cam_vgg, input_image))
    plt.title("VGG Grad-CAM")
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(overlay_heatmap(grad_cam_plus_vgg, input_image))
    plt.title("VGG Grad-CAM++")
    plt.axis('off')

    # SHAP
    plt.subplot(3, 4, 8)
    plt.imshow(shap_heatmap, cmap='jet')
    plt.title("SHAP")
    plt.axis('off')

    # LIME
    plt.subplot(3, 4, 9)
    lime_result = np.squeeze(lime_result)
    marked_image = mark_boundaries(original_image, lime_result, color=(1, 0, 0), mode='overlay')
    plt.imshow(marked_image)
    plt.title("LIME")
    plt.axis('off')

    # Integrated Gradients
    plt.subplot(3, 4, 10)
    plt.imshow(ig_heatmap, cmap='jet')
    plt.axis('off')
    plt.title("Integrated Gradients")

    plt.tight_layout()
    plt.show()

# Select validation images for XAI
def select_val_images(val_loader, num_classes, num_samples_per_class=1):
    class_counts = {i: 0 for i in range(num_classes)}
    selected_images = []
    selected_labels = []
    class_names = full_dataset.classes
    
    for images, labels in val_loader:
        for img, lbl in zip(images, labels):
            lbl = lbl.item()
            if class_counts[lbl] < num_samples_per_class:
                selected_images.append(img)
                selected_labels.append(lbl)
                class_counts[lbl] += 1
                
            if all(count >= num_samples_per_class for count in class_counts.values()):
                break
        else:
            continue
        break
    
    return selected_images, selected_labels, class_names

# Get validation images and run XAI
selected_images, selected_labels, class_names = select_val_images(val_loader, num_classes, num_samples_per_class=1)


# Generate explanations for each class
for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
    input_image = image.unsqueeze(0).to(device)
    print(f"Generating explanations for class: {class_names[label]}")
    generate_and_plot_heatmaps(input_image, model, target_class=label, class_name=class_names[label])