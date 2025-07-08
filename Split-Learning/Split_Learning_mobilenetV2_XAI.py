import multiprocessing
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from captum.attr import LayerGradCam, LayerAttribution
from lime import lime_image
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # ——— Setup ———
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir    = r"Dataset Path"
    batch_size  = 32
    lr          = 1e-3
    num_epochs  = 50
    val_split   = 0.2
    save_best   = True
    client_ckpt = r"Save model path/best_client.pth"
    server_ckpt = r"Save model path/best_server.pth"

    # ——— Data transforms & loaders ———
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    full_ds      = datasets.ImageFolder(root=data_dir, transform=transform)
    classes      = full_ds.classes
    num_classes  = len(classes)
    print(f"Detected classes ({num_classes}): {classes}")

    val_size   = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # ——— Build split MobileNetV2 ———
    mbv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    client_net = nn.Sequential(*list(mbv2.features)).to(device)
    feature_dim = mbv2.classifier[1].in_features  # 1280
    server_net = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(feature_dim, num_classes)
    ).to(device)

    # ——— Loss & optimizers ———
    criterion  = nn.CrossEntropyLoss()
    opt_client = optim.Adam(client_net.parameters(), lr=lr)
    opt_server = optim.Adam(server_net.parameters(), lr=lr)

    history      = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    best_epoch   = -1
    best_val_loss = 0.0
    best_train_loss = 0.0
    best_train_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}:")
        client_net.train(); server_net.train()
        run_loss, run_corr = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = client_net(imgs)
            feats_det = feats.detach().requires_grad_()

            out = server_net(feats_det)
            loss = criterion(out, labels)
            opt_server.zero_grad(); loss.backward(); opt_server.step()

            grad_feats = feats_det.grad
            opt_client.zero_grad(); feats.backward(grad_feats); opt_client.step()

            run_loss += loss.item() * imgs.size(0)
            run_corr += (out.argmax(1) == labels).sum().item()

        train_loss = run_loss / train_size
        train_acc  = run_corr / train_size

        client_net.eval(); server_net.eval()
        val_loss, val_corr = 0.0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = client_net(imgs)
                out = server_net(feats)
                loss = criterion(out, labels)

                val_loss += loss.item() * imgs.size(0)
                val_corr += (out.argmax(1) == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(out, 1).cpu().numpy())

        val_loss = val_loss / val_size
        val_acc  = val_corr / val_size

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, A={val_acc:.4f}\n")

        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_acc = train_acc

            torch.save(client_net.state_dict(), client_ckpt)
            torch.save(server_net.state_dict(), server_ckpt)
            print(f"→ Saved best model (Epoch {epoch}, Val Acc={val_acc:.4f})")

        

    print(f"Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Train L={best_train_loss:.4f}, A={best_train_acc:.4f}\n"
              f"Val   L={best_val_loss:.4f}, A={best_val_acc:.4f}")

    # ——— Plot learning curves ———
    epochs = range(1, num_epochs+1)
    plt.figure(); plt.plot(epochs, history["train_loss"], label="Train Loss"); plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.show()

    plt.figure(); plt.plot(epochs, history["train_acc"], label="Train Acc"); plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.show()

    plt.figure(); plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.grid(True); plt.show()

    plt.figure(); plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(10, 6)); plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(10, 6)); plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.grid(True); plt.show()


    # ——— Load Model ———
    client_net.load_state_dict(torch.load(client_ckpt, map_location=device, weights_only=True))
    server_net.load_state_dict(torch.load(server_ckpt, map_location=device, weights_only=True))
    client_net.eval(); server_net.eval()

    cm_labels, cm_preds, cm_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            feats = client_net(imgs)
            out = server_net(feats)
            probs = torch.softmax(out, dim=1)

            cm_labels.extend(labels.cpu().numpy())
            cm_preds.extend(out.argmax(1).cpu().numpy())
            cm_probs.extend(probs.cpu().numpy())

    # ——— Confusion Matrix ———
    cm = confusion_matrix(cm_labels, cm_preds)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
    print(classification_report(cm_labels, cm_preds, target_names=classes))

    # ——— ROC curves ———
    y_true_bin = label_binarize(cm_labels, classes=list(range(num_classes)))
    y_scores   = np.array(cm_probs)
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right"); plt.grid(True); plt.show()

    # ——— Grad-CAM ———
    full_model = nn.Sequential(client_net, server_net).to(device)
    target_layer = client_net[-1]  # last layer of features
    gradcam = LayerGradCam(full_model, target_layer)

    # grab one sample per class
    samples = {}
    for imgs, labels in val_loader:
        for img, lbl in zip(imgs, labels):
            idx = lbl.item()
            if idx not in samples:
                samples[idx] = img.unsqueeze(0).to(device)
            if len(samples) == num_classes:
                break
        if len(samples) == num_classes:
            break

    # heatmaps
    for idx, cls in enumerate(classes):
        inp = samples[idx]
        attr = gradcam.attribute(inp, target=idx)
        up = LayerAttribution.interpolate(attr, inp.shape[2:])
        heatmap = up[0].cpu().detach().squeeze().numpy()

        img_np = inp[0].cpu().permute(1,2,0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(std * img_np + mean, 0, 1)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        ax1.imshow(img_np); ax1.set_title(f"Original: {cls}"); ax1.axis('off')
        ax2.imshow(img_np, alpha=1.0); ax2.imshow(heatmap, cmap='jet', alpha=0.4)
        ax2.set_title(f"Grad-CAM: {cls}"); ax2.axis('off')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    freeze_support()
    main()
