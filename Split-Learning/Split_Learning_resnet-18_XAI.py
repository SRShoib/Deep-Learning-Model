import multiprocessing
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, f1_score
from sklearn.preprocessing import label_binarize
from captum.attr import LayerGradCam, LayerAttribution
from lime import lime_image
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir    = r"Dataset Path"
    batch_size  = 32
    lr          = 1e-3
    num_epochs  = 50
    val_split   = 0.2
    save_best   = True
    client_ckpt = r"Save model path/best_client.pth"
    server_ckpt = r"Save model path/best_server.pth"

    # Data transforms & loader
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

    # Build split ResNet-18
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    client_net = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3
    ).to(device)
    server_net = nn.Sequential(
        resnet.layer4,
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(resnet.fc.in_features, num_classes)
    ).to(device)

    # Loss & optimizers
    criterion  = nn.CrossEntropyLoss()
    opt_client = optim.Adam(client_net.parameters(), lr=lr)
    opt_server = optim.Adam(server_net.parameters(), lr=lr)

    history      = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    best_epoch   = -1
    best_val_loss = 0.0
    best_train_loss = 0.0
    best_train_acc = 0.0
    

    # Training + validation loop
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        client_net.train(); server_net.train()
        run_loss, run_corr = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            act = client_net(imgs)
            act_det = act.detach().requires_grad_()

            out = server_net(act_det)
            loss = criterion(out, labels)
            opt_server.zero_grad(); loss.backward(); opt_server.step()

            grad_act = act_det.grad
            opt_client.zero_grad(); act.backward(grad_act); opt_client.step()

            run_loss += loss.item() * imgs.size(0)
            preds    = out.argmax(1)
            run_corr += (preds == labels).sum().item()

        train_loss = run_loss / train_size
        train_acc  = run_corr / train_size

        client_net.eval(); server_net.eval()
        val_loss, val_corr = 0.0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                act = client_net(imgs)
                out = server_net(act)
                loss = criterion(out, labels)

                val_loss += loss.item() * imgs.size(0)
                preds    = out.argmax(1)
                val_corr += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(out, 1)[:,1].cpu().numpy())

        val_loss = val_loss / val_size
        val_acc  = val_corr / val_size

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_acc = train_acc

            torch.save(client_net.state_dict(), client_ckpt)
            torch.save(server_net.state_dict(), server_ckpt)
            print(f"→ Saved best model (Epoch {epoch}, Val Acc={val_acc:.4f})")


    print(f"\nBest Val Acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Train Loss={best_train_loss:.4f}, Acc={best_train_acc:.4f}\n"
              f"Val Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}")

    # ——— Plotting ———
    epochs = range(1, num_epochs+1)

    plt.figure(); plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.show()

    plt.figure(); plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
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


    # Load best model
    client_net.load_state_dict(torch.load(client_ckpt, map_location=device, weights_only=True))
    server_net.load_state_dict(torch.load(server_ckpt, map_location=device, weights_only=True))
    client_net.eval(); server_net.eval()

    cm_labels, cm_preds, cm_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            act = client_net(imgs)
            out = server_net(act)
            probs = torch.softmax(out, dim=1)

            cm_labels.extend(labels.cpu().numpy())
            cm_preds.extend(out.argmax(1).cpu().numpy())
            cm_probs.extend(probs.cpu().numpy())

    cm = confusion_matrix(cm_labels, cm_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=full_ds.classes,
        yticklabels=full_ds.classes,
        ax=ax,
        cbar_kws={"shrink": .8}     
    )
    ax.set_title("Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted", labelpad=15)
    ax.set_ylabel("True", labelpad=15)
    # rotate tick labels and align
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    plt.tight_layout()
    plt.show()
    print("\nConfusion Matrix:\n", cm)

    # Classification Report
    print(classification_report(cm_labels, cm_preds, target_names=full_ds.classes))
   
    # Prepare numpy arrays for sklearn metrics
    num_samples = len(cm_labels)
    y_true_bin = label_binarize(cm_labels, classes=range(num_classes))  
    y_scores   = np.array(cm_probs)  # now shape (num_samples, num_classes)

    assert y_scores.shape == (num_samples, num_classes)

    # Plot ROC for each class
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{full_ds.classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi‐Class Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # ————————————— Setup Grad-CAM —————————————
    full_model = nn.Sequential(client_net, server_net).to(device)
    target_layer = client_net[-1]  # last block of layer3
    gradcam = LayerGradCam(full_model, target_layer)

    # ————————————— Grab One Sample per Class —————————————
    samples = {}
    for imgs, labels in val_loader:
        for img, lbl in zip(imgs, labels):
            cls_idx = lbl.item()
            if cls_idx not in samples:
                samples[cls_idx] = img.unsqueeze(0).to(device)
            if len(samples) == num_classes:
                break
        if len(samples) == num_classes:
            break

    # ————————————— Plot Originals & Grad-CAMs —————————————
    for cls_idx, cls_name in enumerate(classes):
        inp = samples[cls_idx]               
        # compute attribution
        attr = gradcam.attribute(inp, target=cls_idx)  
        up   = LayerAttribution.interpolate(attr, inp.shape[2:])
        heatmap = up[0].cpu().detach().squeeze().numpy()

        # undo normalization for display
        img_np = inp[0].cpu().permute(1,2,0).numpy()
        mean   = np.array([0.485, 0.456, 0.406])
        std    = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # heatmap overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_np)
        ax1.set_title(f"Original: {cls_name}")
        ax1.axis('off')

        ax2.imshow(img_np, alpha=1.0)
        ax2.imshow(heatmap, cmap='jet', alpha=0.4, interpolation='bilinear')
        ax2.set_title(f"Grad-CAM: {cls_name}")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    freeze_support()
    main()
