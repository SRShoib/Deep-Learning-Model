import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import shap
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_path = r"Dataset path"  # Path to your dataset
model_save_path = r"model save location\first.pth" # best model save location

# Configurations
batch_size = 32
image_size = 224
num_epochs = 50
validation_split = 0.8


# Data Preparation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# 80/20 split
train_size = int(validation_split * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Fusion Model
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)

        self.fusion = nn.Sequential(
            nn.Linear(768 + 768 + 384, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)
        swin_feat = self.swin(x)
        deit_feat = self.deit(x)
        fused = torch.cat([vit_feat, swin_feat, deit_feat], dim=1)
        return self.fusion(fused)

model = FusionModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training + Evaluation
best_val_acc = 0
train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)

    model.eval()
    correct, total = 0, 0
    val_loss, all_preds, all_labels, all_probs = 0, [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            val_loss += loss.item()
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / len(val_loader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved at {model_save_path}")


# Final output
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load best model
model = FusionModel(num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

# Collect predictions and probabilities
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# Binarize the labels for ROC curves
y_true_bin = label_binarize(all_labels, classes=range(num_classes))
y_scores = np.array(all_probs)

# Plot ROC Curves
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

# Accuracy/Loss Curve
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.title("Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.show()


# Visualization Utilities

# Grad-CAM (standard, similar to Grad-CAM++)
def grad_cam(model, input_tensor):
    fmap, grad = None, None

    def forward_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    # Register last block
    if hasattr(model, 'blocks'):
        handle_f = model.blocks[-1].register_forward_hook(forward_hook)
        handle_b = model.blocks[-1].register_full_backward_hook(backward_hook)
    elif hasattr(model, 'layers'):
        handle_f = model.layers[-1].blocks[-1].register_forward_hook(forward_hook)
        handle_b = model.layers[-1].blocks[-1].register_full_backward_hook(backward_hook)

    model.eval()
    output = model(input_tensor)
    pred_class = output.argmax(1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    cam = torch.sum(fmap * grad, dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.nan_to_num(cam)

    if cam.ndim == 1:
        cam = cam[np.newaxis, :]  # shape: (1, N)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    handle_f.remove()
    handle_b.remove()
    return overlay

def grad_cam_plus(model, input_tensor):

    fmap, grad = None, None

    def forward_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    # Register hooks based on model type
    if hasattr(model, 'blocks'):  # ViT or DeiT
        handle_f = model.blocks[-1].register_forward_hook(forward_hook)
        handle_b = model.blocks[-1].register_full_backward_hook(backward_hook)
    elif hasattr(model, 'layers'):  # Swin
        handle_f = model.layers[-1].blocks[-1].register_forward_hook(forward_hook)
        handle_b = model.layers[-1].blocks[-1].register_full_backward_hook(backward_hook)
    else:
        raise ValueError("Unsupported model type for Grad-CAM++")

    # Forward and backward pass
    model.eval()
    output = model(input_tensor)
    pred_class = output.argmax(1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    # Get Grad-CAM map
    weights = grad.mean(dim=1, keepdim=True)
    cam = torch.sum(weights * fmap, dim=2).squeeze().cpu().numpy()  # shape: [tokens]

    # Remove CLS token for ViT/DeiT
    if hasattr(model, 'blocks') and cam.shape[0] in [197, 577]:
        cam = cam[1:]

    # Normalize CAM
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.nan_to_num(cam)

    # Reshape to square if possible
    token_count = cam.shape[0]
    sqrt_token = int(np.sqrt(token_count))
    if sqrt_token * sqrt_token == token_count:
        cam = cam.reshape(sqrt_token, sqrt_token)
    else:
        cam = cam.reshape(1, -1)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Resize CAM to match input image size
    if cam.shape != (224, 224):
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Prepare input image for overlay
    img_np = input_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    return overlay

# LIME
def lime_explanation(model, input_tensor):
    def batch_predict(images):
        model.eval()
        batch = torch.tensor(images.transpose((0, 3, 1, 2))).float()
        batch = (batch / 255.0 - 0.5) / 0.5
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
        return F.softmax(logits, dim=1).cpu().numpy()

    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(img, batch_predict, top_labels=1, hide_color=0, num_samples=1000)
    lime_img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)
    return mark_boundaries(lime_img, mask)


def shap_explanation(model, input_tensor):

    # Convert image tensor to numpy
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)

    # Define model prediction wrapper
    def model_predict(images):
        images = torch.tensor(images.transpose((0, 3, 1, 2))).float()
        images = (images / 255.0 - 0.5) / 0.5
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        return F.softmax(logits, dim=1).cpu().numpy()

    # SHAP Masker for 224x224 vision input
    masker = shap.maskers.Image("blur(128,128)", img.shape)

    # Build SHAP explainer
    explainer = shap.Explainer(model_predict, masker, output_names=class_names)

    # Get SHAP values for a single input image
    shap_values = explainer(np.expand_dims(img, axis=0), max_evals=200, batch_size=20)

    # Extract mean SHAP heatmap across channels
    shap_img = shap_values[0].values.mean(axis=-1)
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
    shap_img = np.nan_to_num(shap_img)

    # Resize and colorize
    shap_img = cv2.resize(shap_img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(shap_img * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay




def attention_rollout(model, input_tensor):
    attn_weights = []

    def attn_hook(module, input, output):
        attn_weights.append(output.detach())

    handles = []
    if hasattr(model, 'blocks'):  # ViT / DeiT
        for block in model.blocks:
            handles.append(block.attn.register_forward_hook(attn_hook))
        use_cls = True
    elif hasattr(model, 'layers'):  # Swin
        for layer in model.layers:
            for block in layer.blocks:
                handles.append(block.attn.register_forward_hook(attn_hook))
        use_cls = False

    _ = model(input_tensor)

    if use_cls:
        # Standard attention rollout with CLS token
        result = torch.eye(attn_weights[0].size(-1)).to(input_tensor.device)
        for attn in attn_weights:
            attn_heads = attn.mean(dim=1)
            attn_heads = attn_heads + torch.eye(attn_heads.size(-1)).to(input_tensor.device)
            attn_heads /= attn_heads.sum(dim=-1, keepdim=True)
            result = attn_heads @ result
        rollout = result[0, 1:]  # remove CLS token
    else:
        # Swin fallback
        attn = attn_weights[-1]  # [B, H, N, N]
        attn = attn.mean(dim=1)[0]  # average heads: [N, N]
        rollout = attn.mean(dim=0)  # average attention received: [N]

        if rollout.ndim == 0:  # fix scalar case
            rollout = rollout.unsqueeze(0)

        rollout = rollout.cpu().numpy()

        if rollout.size == 0 or np.isnan(rollout).any():
            rollout = np.ones((14, 14))  # fallback to neutral overlay
        else:
            num_patches = rollout.shape[0]
            side = int(np.ceil(np.sqrt(num_patches)))
            pad_len = side * side - num_patches
            rollout = np.pad(rollout, (0, pad_len), mode='constant', constant_values=0)
            rollout = rollout.reshape(side, side)

        rollout_map = rollout
    if use_cls:
        # Post-process rollout from CLS-based models
        rollout = rollout.cpu().numpy()
        num_patches = rollout.shape[0]
        side = int(np.ceil(np.sqrt(num_patches)))
        pad_len = side * side - num_patches
        rollout = np.pad(rollout, (0, pad_len), mode='constant', constant_values=0)
        rollout_map = rollout.reshape(side, side)

    # Normalize safely
    if rollout_map.max() != rollout_map.min():
        rollout_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min())
    else:
        rollout_map = np.zeros_like(rollout_map)
    rollout_map = np.nan_to_num(rollout_map, nan=0.0)
    rollout_map = cv2.resize(rollout_map, (224, 224))

    # Overlay on image
    img_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = img_np.astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(rollout_map * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    for h in handles:
        h.remove()
    return overlay





# -----------------------------
# Generate Visualizations
# -----------------------------

# Load best model
model = FusionModel(num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

vit = model.vit.to(device)
swin = model.swin.to(device)
deit = model.deit.to(device)

# Create mapping of class index to example image
example_per_class = {}

# Search val_loader to find one sample per class
with torch.no_grad():
    for images, labels in val_loader:
        for img, label in zip(images, labels):
            cls = label.item()
            if cls not in example_per_class:
                example_per_class[cls] = img.unsqueeze(0).to(device)
            if len(example_per_class) == num_classes:
                break
        if len(example_per_class) == num_classes:
            break

# Now generate visualizations for each class
for cls_idx, input_tensor in example_per_class.items():
    class_name = class_names[cls_idx]
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    axes[0,0].imshow(attention_rollout(vit, input_tensor))
    axes[0,0].set_title(f"ViT Attention Rollout - {class_name}")
    axes[0,1].imshow(grad_cam_plus(vit, input_tensor))
    axes[0,1].set_title(f"ViT Grad-CAM++ - {class_name}")

    axes[1,0].imshow(attention_rollout(swin, input_tensor))
    axes[1,0].set_title(f"Swin Attention Rollout - {class_name}")
    axes[1,1].imshow(grad_cam_plus(swin, input_tensor))
    axes[1,1].set_title(f"Swin Grad-CAM++ - {class_name}")

    axes[2,0].imshow(attention_rollout(deit, input_tensor))
    axes[2,0].set_title(f"DeiT Attention Rollout - {class_name}")
    axes[2,1].imshow(grad_cam_plus(deit, input_tensor))
    axes[2,1].set_title(f"DeiT Grad-CAM++ - {class_name}")

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Class: {class_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# -----------------------------
# Combined Visualizations
# -----------------------------
# Reuse earlier val_loader logic
for cls_idx, input_tensor in example_per_class.items():
    class_name = class_names[cls_idx]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    models_dict = {'ViT': vit, 'Swin': swin, 'DeiT': deit}

    for i, (name, model_part) in enumerate(models_dict.items()):
        axes[i, 0].imshow(grad_cam(model_part, input_tensor))
        axes[i, 0].set_title(f"{name} Grad-CAM")

        axes[i, 1].imshow(grad_cam_plus(model_part, input_tensor))
        axes[i, 1].set_title(f"{name} Grad-CAM++")

        axes[i, 2].imshow(attention_rollout(model_part, input_tensor))
        axes[i, 2].set_title(f"{name} Rollout")

        axes[i, 3].imshow(lime_explanation(model_part, input_tensor))
        axes[i, 3].set_title(f"{name} LIME")

        # axes[i, 4].imshow(shap_explanation(model_part, input_tensor))
        # axes[i, 4].set_title(f"{name} SHAP")

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'All XAI Methods - Class: {class_name}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()