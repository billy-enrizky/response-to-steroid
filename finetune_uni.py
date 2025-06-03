import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import pickle
from collections import defaultdict
import gc

# --- Configuration OPTIMIZED for GPU Memory ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4
BATCH_SIZE = 1  # Reduced from 32 to 1 for memory efficiency
NUM_EPOCHS = 12
NUM_CLASSES = 2
MODEL_NAME = "uni2-h"
SEED = 42

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths to your data
TRAIN_RESPONSE_PATH = "patches_train_response_portal_tract"
TRAIN_NO_RESPONSE_PATH = "patches_train_no_response_portal_tract"
TEST_RESPONSE_PATH = "patches_test_response_portal_tract"
TEST_NO_RESPONSE_PATH = "patches_test_no_response_portal_tract"

print(f"Memory-optimized configuration:")
print(f"- Learning Rate: {LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE} (reduced for memory)")
print(f"- Epochs: {NUM_EPOCHS}")
print(f"- Device: {DEVICE}")

# --- 1. Dataset and DataLoader ---
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.data = list(zip(image_paths, labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def collect_images_and_labels(folder, label):
    return [(os.path.join(folder, f), label)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Collect training images
train_images_data = (collect_images_and_labels(TRAIN_RESPONSE_PATH, 0) +
                     collect_images_and_labels(TRAIN_NO_RESPONSE_PATH, 1))

# Collect test images
test_images_data = (collect_images_and_labels(TEST_RESPONSE_PATH, 0) +
                    collect_images_and_labels(TEST_NO_RESPONSE_PATH, 1))

print(f"Training data: {len(train_images_data)} patches")
print(f"Test data: {len(test_images_data)} patches")

# Extract paths and labels
train_img_paths = [item[0] for item in train_images_data]
train_labels = [item[1] for item in train_images_data]
test_img_paths = [item[0] for item in test_images_data]
test_labels_gt = [item[1] for item in test_images_data]

# Split training data into train and validation (80-20 split)
train_paths, val_paths, train_lbls, val_lbls = train_test_split(
    train_img_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_img_paths)}")

# --- Create and save index CSV files ---
def make_df(image_list, data_type):
    """Convert list[(path, label)] → DataFrame."""
    img_paths = [p for p, _ in image_list]
    labels = [l for _, l in image_list]
    classes = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({
        "image_path": img_paths,
        "label": labels,
        "class_name": classes,
        "data_type": data_type
    })

# Create full dataset DataFrame
df_train = make_df(train_images_data, "train")
df_test = make_df(test_images_data, "test")
df_full = pd.concat([df_train, df_test], ignore_index=True)

# Save index path CSV
csv_out = "index_path_separated_portal_tract.csv"
df_full.to_csv(csv_out, index_label="index")
print(f"Saved image-path index to {csv_out}")

# --- 2. Load UNI Model ---
try:
    from uni import get_encoder
    base_model, model_transform = get_encoder(enc_name=MODEL_NAME, device=DEVICE)
    print(f"Loaded '{MODEL_NAME}' model and its specific transform.")
    
    # Clear memory after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error loading UNI model: {e}")
    exit()

# Determine feature dimension
try:
    base_model.eval()
    with torch.no_grad():
        dummy_image = Image.new('RGB', (224, 224))
        dummy_input = model_transform(dummy_image).unsqueeze(0).to(DEVICE)
        features = base_model(dummy_input)
        
        if isinstance(features, torch.Tensor):
            if features.ndim == 4:  # Conv feature maps [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            elif features.ndim == 3:  # ViT-style [B, N_patches, D]
                if features.shape[1] > 1:
                    features = features[:, 0]  # Use CLS token

    feature_dim = features.shape[-1]
    print(f"Detected feature dimension: {feature_dim}")
    
    # Clear dummy tensors
    del dummy_input, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Could not determine feature dimension: {e}")
    feature_dim = 1536  # Default for uni2-h
    print(f"Using default feature dimension: {feature_dim}")

# --- 3. Memory-Efficient Fine-tuned Model Architecture ---
class FineTunedUNImodel(nn.Module):
    def __init__(self, base_uni_model, num_features, num_classes_out):
        super().__init__()
        self.base_model = base_uni_model
        self.dropout = nn.Dropout(0.3)  # Reduced dropout for memory
        # Simplified classifier for memory efficiency
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),  # Smaller hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes_out)
        )

    def forward(self, x):
        with torch.cuda.amp.autocast():  # Mixed precision for memory efficiency
            features = self.base_model(x)
            
            if isinstance(features, torch.Tensor):
                if features.ndim == 4:  # Conv feature maps
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                elif features.ndim == 3:  # ViT style
                    if features.shape[1] > 1:
                        features = features[:, 0]  # Use CLS token
                    else:
                        features = features.squeeze(1)
            
            features = self.dropout(features)
            output = self.classifier(features)
            return output

# Instantiate model
model_to_finetune = FineTunedUNImodel(base_model, feature_dim, NUM_CLASSES)
model_to_finetune = model_to_finetune.to(DEVICE)

# --- 4. Memory-Efficient Data Loaders ---
train_dataset = CustomImageDataset(train_paths, train_lbls, transform=model_transform)
val_dataset = CustomImageDataset(val_paths, val_lbls, transform=model_transform)
test_dataset = CustomImageDataset(test_img_paths, test_labels_gt, transform=model_transform)

# Reduced num_workers and disabled pin_memory for memory efficiency
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=2, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=2, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=2, pin_memory=False)

print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")

# --- 5. Training Setup with Memory Optimization ---
# Start with frozen backbone
for param in model_to_finetune.base_model.parameters():
    param.requires_grad = False

# Initial optimizer for classifier only
optimizer = optim.AdamW(
    model_to_finetune.classifier.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=2, factor=0.5, verbose=True
)

# Mixed precision scaler for memory efficiency
scaler = torch.cuda.amp.GradScaler()

# --- 6. Memory-Efficient Training Loop ---
best_val_accuracy = 0.0
best_model_path = "finetuned_uni_model_best.pth"
patience_counter = 0
max_patience = 4

print(f"\nStarting memory-efficient fine-tuning for {NUM_EPOCHS} epochs on {DEVICE}...")
print(f"Learning Rate: {LEARNING_RATE} (batch size: {BATCH_SIZE})")

for epoch in range(NUM_EPOCHS):
    # Unfreeze backbone after 4 epochs (delayed due to smaller batch size)
    if epoch == 4:
        print(f"\nUnfreezing backbone for full fine-tuning...")
        for param in model_to_finetune.base_model.parameters():
            param.requires_grad = True
        
        # Differential learning rates
        backbone_lr = LEARNING_RATE / 20  # 2.5e-5 for backbone
        classifier_lr = LEARNING_RATE     # 5e-4 for classifier
        
        optimizer = optim.AdamW([
            {'params': model_to_finetune.base_model.parameters(), 'lr': backbone_lr},
            {'params': model_to_finetune.classifier.parameters(), 'lr': classifier_lr}
        ], weight_decay=0.01, betas=(0.9, 0.999))
        
        # Update scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2, factor=0.5, verbose=True
        )
        
        print(f"Backbone LR: {backbone_lr}, Classifier LR: {classifier_lr}")
    
    # Training phase
    model_to_finetune.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model_to_finetune(images)
            loss = criterion(outputs, labels)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_to_finetune.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(
            loss=loss.item(), 
            acc=100.*correct_train/total_train if total_train > 0 else 0,
            lr=f"{current_lr:.1e}"
        )
        
        # Clear cache every 20 batches
        if batch_idx % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc_train = 100. * correct_train / len(train_loader.dataset)

    # Validation phase
    model_to_finetune.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model_to_finetune(images)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_acc_val = 100. * correct_val / len(val_loader.dataset)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_acc_val:.2f}%, LR: {current_lr:.1e}")

    # Step scheduler
    scheduler.step(epoch_val_loss)

    # Save best model with early stopping
    if epoch_acc_val > best_val_accuracy:
        best_val_accuracy = epoch_acc_val
        patience_counter = 0
        torch.save({
            'model_state_dict': model_to_finetune.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_accuracy': best_val_accuracy,
            'feature_dim': feature_dim,
            'num_classes': NUM_CLASSES,
            'learning_rate': LEARNING_RATE
        }, best_model_path)
        print(f"✓ Saved new best model (Val Acc: {best_val_accuracy:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= max_patience and epoch > 6:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Clear cache after each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

print(f"\nFine-tuning finished. Best validation accuracy: {best_val_accuracy:.2f}%")

# --- 7. Memory-Efficient Testing ---
print("\nLoading best model for testing...")
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model_to_finetune.load_state_dict(checkpoint['model_state_dict'])
model_to_finetune.eval()

# Clear cache before testing
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n--- Testing Fine-tuned Model ---")
test_loss = 0.0
correct_test = 0
total_test = 0
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model_to_finetune(images)
            loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        
        # Clear cache every 10 batches during testing
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

test_accuracy = 100. * correct_test / total_test
test_loss_avg = test_loss / len(test_loader.dataset)
print(f"Fine-tuned Model Test Accuracy: {test_accuracy:.2f}%")
print(f"Fine-tuned Model Test Loss: {test_loss_avg:.4f}")

# --- 8. Extract Features (Memory-Efficient) ---
print("\n--- Extracting Features for Evaluation ---")

# Free up model memory temporarily
del model_to_finetune
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

# Create smaller batch size loaders for feature extraction
feature_batch_size = 4  # Even smaller for feature extraction
train_dataloader_for_features = DataLoader(train_dataset, batch_size=feature_batch_size, 
                                         shuffle=False, num_workers=1, pin_memory=False)
test_dataloader_for_features = DataLoader(test_dataset, batch_size=feature_batch_size, 
                                        shuffle=False, num_workers=1, pin_memory=False)

print("Extracting training features...")
train_features = extract_patch_features_from_dataloader(base_model, train_dataloader_for_features)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
print("Extracting test features...")
test_features = extract_patch_features_from_dataloader(base_model, test_dataloader_for_features)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Convert to tensors
train_feats = torch.Tensor(train_features['embeddings'])
train_labels_tensor = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels_tensor = torch.Tensor(test_features['labels']).type(torch.long)

print(f"Train features shape: {train_feats.shape}, Test features shape: {test_feats.shape}")

# Clear base model from GPU memory for sklearn evaluations
base_model.cpu()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- 8. Evaluation Functions ---
def get_eval_metrics(targets_all, preds_all, probs_all=None, prefix=""):
    targets_all = np.asarray(targets_all)
    preds_all = np.asarray(preds_all)
    
    acc = accuracy_score(targets_all, preds_all)
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
    
    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
        f"{prefix}report": cls_rep
    }
    
    if probs_all is not None:
        probs_all = np.asarray(probs_all)
        if len(np.unique(targets_all)) > 1:
            try:
                if probs_all.ndim == 2 and probs_all.shape[1] == 2:
                    roc_auc = roc_auc_score(targets_all, probs_all[:, 1])
                else:
                    roc_auc = roc_auc_score(targets_all, probs_all)
                eval_metrics[f"{prefix}auroc"] = roc_auc
            except:
                eval_metrics[f"{prefix}auroc"] = np.nan
    
    return eval_metrics

def eval_sklearn_classifier(classifier, train_feats, train_labels, test_feats, test_labels, 
                           prefix="", scale_features=False):
    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_feats.cpu().numpy()
    y_test = test_labels.cpu().numpy()
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    classifier.fit(X_train, y_train)
    preds_all = classifier.predict(X_test)
    
    probs_all = None
    if hasattr(classifier, "predict_proba"):
        try:
            probs_all = classifier.predict_proba(X_test)
        except:
            pass
    
    metrics = get_eval_metrics(y_test, preds_all, probs_all, prefix=prefix)
    
    return metrics, {
        "preds_all": preds_all,
        "probs_all": probs_all,
        "targets_all": y_test,
        "classifier": classifier,
        "scaler": scaler
    }

# --- 9. Baseline Method Evaluations ---
print("\n=== BASELINE METHOD EVALUATIONS ===")

# Linear Probe
try:
    from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
    print("\n--- Linear Probe Evaluation ---")
    linprobe_metrics, linprobe_dump = eval_linear_probe(
        train_feats=train_feats, train_labels=train_labels_tensor,
        valid_feats=None, valid_labels=None,
        test_feats=test_feats, test_labels=test_labels_tensor,
        max_iter=1000, verbose=True
    )
    print(f"Linear Probe Accuracy: {linprobe_metrics.get('lin_acc', 0):.4f}")
except Exception as e:
    print(f"Linear Probe evaluation failed: {e}")
    linprobe_metrics = {"lin_acc": 0}

# KNN (UNI implementation)
try:
    from uni.downstream.eval_patch_features.fewshot import eval_knn
    print("\n--- KNN (UNI) Evaluation ---")
    knn_metrics, knn_dump, proto_metrics, proto_dump = eval_knn(
        train_feats=train_feats, train_labels=train_labels_tensor,
        test_feats=test_feats, test_labels=test_labels_tensor,
        center_feats=True, normalize_feats=True, n_neighbors=5  # Reduced for small dataset
    )
    print(f"KNN (UNI) Accuracy: {knn_metrics.get('knn5_acc', 0):.4f}")
except Exception as e:
    print(f"KNN (UNI) evaluation failed: {e}")
    knn_metrics = {"knn5_acc": 0}

# SVM with optimal parameters for small dataset
print("\n--- SVM Evaluation ---")
svm_classifier = SVC(
    C=0.1,  # Less regularization for small dataset
    kernel='rbf', 
    gamma='scale', 
    class_weight='balanced', 
    probability=True, 
    random_state=SEED
)
svm_metrics, svm_dump = eval_sklearn_classifier(
    classifier=svm_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="svm_", scale_features=True
)
print(f"SVM Accuracy: {svm_metrics.get('svm_acc', 0):.4f}")

# Random Forest optimized for small dataset
print("\n--- Random Forest Evaluation ---")
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Reduced for small dataset
    max_depth=10,      # Shallower trees
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight='balanced', 
    random_state=SEED, 
    n_jobs=-1
)
rf_metrics, rf_dump = eval_sklearn_classifier(
    classifier=rf_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="rf_", scale_features=False
)
print(f"Random Forest Accuracy: {rf_metrics.get('rf_acc', 0):.4f}")

# Gradient Boosting optimized for small dataset
print("\n--- Gradient Boosting Evaluation ---")
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,     # Reduced for small dataset
    learning_rate=0.05,   # Lower learning rate
    max_depth=4,          # Shallower trees
    subsample=0.8,
    random_state=SEED
)
gb_metrics, gb_dump = eval_sklearn_classifier(
    classifier=gb_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="gb_", scale_features=False
)
print(f"Gradient Boosting Accuracy: {gb_metrics.get('gb_acc', 0):.4f}")

# KNN (sklearn) optimized for small dataset
print("\n--- KNN (sklearn) Evaluation ---")
knn_sklearn = KNeighborsClassifier(
    n_neighbors=5,  # Reduced for small dataset
    weights='distance', 
    metric='euclidean', 
    n_jobs=-1
)
knn_sklearn_metrics, knn_sklearn_dump = eval_sklearn_classifier(
    classifier=knn_sklearn,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="knn_sklearn_", scale_features=True
)
print(f"KNN (sklearn) Accuracy: {knn_sklearn_metrics.get('knn_sklearn_acc', 0):.4f}")

# --- 10. Find Best Model ---
print("\n--- Finding Best Model by Accuracy ---")

results_summary = {
    "Fine-tuned UNI": test_accuracy / 100,
    "Linear Probe": linprobe_metrics.get('lin_acc', 0),
    "KNN (UNI)": knn_metrics.get('knn5_acc', 0),
    "SVM": svm_metrics.get('svm_acc', 0),
    "Random Forest": rf_metrics.get('rf_acc', 0),
    "Gradient Boosting": gb_metrics.get('gb_acc', 0),
    "KNN (sklearn)": knn_sklearn_metrics.get('knn_sklearn_acc', 0)
}

# Sort results by accuracy
sorted_results = sorted(results_summary.items(), key=lambda x: x[1], reverse=True)
best_model_name, best_accuracy = sorted_results[0]

print(f"🏆 Best performing model: {best_model_name} with {best_accuracy:.4f} accuracy")

# Get best model predictions and probabilities
if best_model_name == "Fine-tuned UNI":
    best_predictions = np.array(all_predictions)
    best_probabilities = np.array(all_probabilities)
elif best_model_name == "SVM":
    best_predictions = svm_dump["preds_all"]
    best_probabilities = svm_dump["probs_all"]
elif best_model_name == "Random Forest":
    best_predictions = rf_dump["preds_all"]
    best_probabilities = rf_dump["probs_all"]
elif best_model_name == "Gradient Boosting":
    best_predictions = gb_dump["preds_all"]
    best_probabilities = gb_dump["probs_all"]
elif best_model_name == "KNN (sklearn)":
    best_predictions = knn_sklearn_dump["preds_all"]
    best_probabilities = knn_sklearn_dump["probs_all"]
else:
    # Fallback to fine-tuned model
    best_predictions = np.array(all_predictions)
    best_probabilities = np.array(all_probabilities)

# --- 11. Create Test Images DataFrame ---
print("\n--- Creating Test Images DataFrame ---")
test_imgs_df = pd.DataFrame(test_dataset.data, columns=['path', 'label'])
test_imgs_df.to_csv('test_imgs_df_portal_tract_separated.csv', index=False)
print("Saved test_imgs_df_portal_tract_separated.csv")

# --- 12. Top-k Patch Retrieval using Best Model ---
print(f"\n--- Top-k Patch Retrieval using Best Model ({best_model_name}) ---")

# Calculate confidence scores for ranking
if best_probabilities is not None:
    # For binary classification, use probability of each class
    if best_probabilities.ndim == 2 and best_probabilities.shape[1] == 2:
        confidence_response = best_probabilities[:, 0]      # Probability of class 0 (response)
        confidence_no_response = best_probabilities[:, 1]   # Probability of class 1 (no response)
    else:
        # Fallback to predictions with random confidence
        confidence_response = (1 - best_predictions) + np.random.rand(len(best_predictions)) * 0.1
        confidence_no_response = best_predictions + np.random.rand(len(best_predictions)) * 0.1
else:
    # No probabilities available, use predictions with random confidence
    print("No probabilities available, using prediction-based confidence")
    confidence_response = (1 - best_predictions) + np.random.rand(len(best_predictions)) * 0.1
    confidence_no_response = best_predictions + np.random.rand(len(best_predictions)) * 0.1

# Get all indices sorted by confidence for each class
response_indices = np.argsort(confidence_response)[::-1]      # Highest confidence first
no_response_indices = np.argsort(confidence_no_response)[::-1]  # Highest confidence first

# Import concat_images for concatenating images
try:
    from uni.downstream.utils import concat_images
except ImportError:
    print("Warning: concat_images not available, will save indices only")
    concat_images = None

# Function to save top-k patches
def save_topk_patches(indices, class_name, k_values, confidence_scores):
    # Save all top-k indices to CSV
    df_indices = pd.DataFrame({
        'index': indices[:max(k_values)],
        'confidence': confidence_scores[indices[:max(k_values)]]
    })
    csv_path = f"topk_{class_name}_indices_portal_tract_separated.csv"
    df_indices.to_csv(csv_path, index=False)
    print(f"Saved top-k indices for {class_name} to {csv_path}")
    
    # Generate concatenated images for different k values
    if concat_images is not None:
        for k in k_values:
            if len(indices) >= k:
                topk_indices = indices[:k]
                images = [Image.open(test_imgs_df['path'].iloc[idx]) for idx in topk_indices]
                concat_img = concat_images(images, gap=5)
                save_path = f"Top{k}_{class_name}_portal_tract_separated_{best_model_name.replace(' ', '_')}.png"
                concat_img.save(save_path)
                print(f"Saved concatenated top {k} {class_name} image to {save_path}")
                
                # Also save individual top-k indices CSV
                df_topk = pd.DataFrame({
                    'index': topk_indices,
                    'confidence': confidence_scores[topk_indices]
                })
                csv_topk_path = f"top{k}_{class_name}_indices_portal_tract_separated.csv"
                df_topk.to_csv(csv_topk_path, index=False)

# Save top-k patches for both classes
k_values = [5, 10, 15, 20, 25]
print("\n--- Top-k Response Patches ---")
save_topk_patches(response_indices, "response", k_values, confidence_response)

print("\n--- Top-k No Response Patches ---")
save_topk_patches(no_response_indices, "no_response", k_values, confidence_no_response)

# --- 13. Slide-level Evaluation using Best Model ---
print(f"\n--- Slide-level Evaluation using Best Model ({best_model_name}) ---")

# Dictionary to collect patch predictions for each slide
slide_preds = defaultdict(list)
slide_actual = {}

# Iterate through the test dataset and corresponding patch predictions
for (img_path, actual_label), pred_label in zip(test_dataset.data, best_predictions):
    basename = os.path.basename(img_path)
    parts = basename.split('-')
    if len(parts) < 2:
        print(f"Warning: Filename format unexpected: {basename}")
        continue
    
    # The slide id is the biopsy patient id (second element)
    slide_id = parts[1]
    
    # Store prediction for this slide
    slide_preds[slide_id].append(pred_label)
    
    # Set the actual label for the slide
    if slide_id not in slide_actual:
        slide_actual[slide_id] = actual_label
    else:
        if slide_actual[slide_id] != actual_label:
            print(f"Warning: inconsistent actual labels for slide {slide_id}")

# Compute majority vote prediction for each slide
slide_pred_majority = {}
slide_pred_confidence = {}
for slide, preds in slide_preds.items():
    # Majority vote: if average is less than 0.5 then label 0; otherwise label 1
    avg_pred = np.mean(preds)
    majority_label = 0 if avg_pred < 0.5 else 1
    slide_pred_majority[slide] = majority_label
    slide_pred_confidence[slide] = abs(avg_pred - 0.5) * 2  # Confidence measure

# Calculate slide-level accuracy
y_true_slides = []
y_pred_slides = []
slide_confidences = []
for slide in slide_actual:
    y_true_slides.append(slide_actual[slide])
    y_pred_slides.append(slide_pred_majority.get(slide, 0))
    slide_confidences.append(slide_pred_confidence.get(slide, 0))

slide_accuracy = np.mean(np.array(y_true_slides) == np.array(y_pred_slides))
slide_bacc = balanced_accuracy_score(y_true_slides, y_pred_slides)

print(f"\nSlide-level Evaluation Metrics (Best Model: {best_model_name}):")
print(f"Slide-level Accuracy: {slide_accuracy:.4f}")
print(f"Slide-level Balanced Accuracy: {slide_bacc:.4f}")

# Compute and log the confusion matrix
cm = confusion_matrix(y_true_slides, y_pred_slides)
print("Slide-level Confusion Matrix (rows: actual, columns: predicted):")
print(cm)

# Additional slide-level metrics
slide_metrics = get_eval_metrics(y_true_slides, y_pred_slides, get_report=True, prefix="slide_")
print("Detailed Slide-level Metrics:")
for k, v in slide_metrics.items():
    if "report" not in k:
        print(f"  {k}: {v:.4f}")

# Save slide-level results
slide_results_df = pd.DataFrame({
    'slide_id': list(slide_actual.keys()),
    'actual_label': [slide_actual[sid] for sid in slide_actual.keys()],
    'predicted_label': [slide_pred_majority.get(sid, 0) for sid in slide_actual.keys()],
    'confidence': [slide_pred_confidence.get(sid, 0) for sid in slide_actual.keys()],
    'num_patches': [len(slide_preds[sid]) for sid in slide_actual.keys()],
    'correct_prediction': [slide_actual[sid] == slide_pred_majority.get(sid, 0) for sid in slide_actual.keys()]
})
slide_results_path = f"slide_level_results_portal_tract_separated_{best_model_name.replace(' ', '_')}.csv"
slide_results_df.to_csv(slide_results_path, index=False)
print(f"Saved slide-level results to {slide_results_path}")

# --- 14. Save All Results ---
print("\n--- Saving All Results ---")

# Save fine-tuned model predictions
finetuned_results = {
    'predictions': all_predictions,
    'labels': all_labels,
    'probabilities': all_probabilities,
    'test_accuracy': test_accuracy,
    'test_loss': test_loss_avg,
    'best_val_accuracy': best_val_accuracy,
    'learning_rate_used': LEARNING_RATE,
    'epochs_trained': checkpoint['epoch'] + 1
}

with open('finetuned_uni_test_results.pkl', 'wb') as f:
    pickle.dump(finetuned_results, f)

# Save all results summary
all_results = {
    'finetuned_uni': finetuned_results,
    'linear_probe': linprobe_metrics,
    'knn_uni': knn_metrics,
    'svm': svm_metrics,
    'random_forest': rf_metrics,
    'gradient_boosting': gb_metrics,
    'knn_sklearn': knn_sklearn_metrics,
    'summary': results_summary,
    'best_model': {
        'name': best_model_name,
        'accuracy': best_accuracy,
        'predictions': best_predictions.tolist(),
        'probabilities': best_probabilities.tolist() if best_probabilities is not None else None
    },
    'slide_level_results': {
        'accuracy': slide_accuracy,
        'balanced_accuracy': slide_bacc,
        'confusion_matrix': cm.tolist(),
        'metrics': slide_metrics
    },
    'configuration': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'dataset_size': len(train_paths)
    }
}

with open('all_evaluation_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Create results DataFrame
results_df = pd.DataFrame([
    {'Model': name, 'Accuracy': acc, 'Percentage': f"{acc*100:.2f}%"}
    for name, acc in sorted_results
])
results_df.to_csv('model_comparison_results.csv', index=False)

print("Results saved to:")
print("- index_path_separated_portal_tract.csv")
print("- test_imgs_df_portal_tract_separated.csv")
print("- finetuned_uni_test_results.pkl")
print("- all_evaluation_results.pkl") 
print("- model_comparison_results.csv")
print(f"- slide_level_results_portal_tract_separated_{best_model_name.replace(' ', '_')}.csv")

print(f"\n🎯 Evaluation complete! Best model: {best_model_name} ({best_accuracy*100:.2f}%)")
print(f"📊 Slide-level accuracy: {slide_accuracy*100:.2f}%")
print(f"📈 Learning rate strategy (5e-4) optimized for {len(train_paths)} training samples")