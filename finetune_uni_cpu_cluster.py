import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
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
import logging
import time
from datetime import datetime

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/cluster/projects/bhatgroup/response_to_steroid/finetune_uni_cpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Force CPU-Only Configuration ---
# Disable CUDA completely
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

DEVICE = torch.device("cpu")
LEARNING_RATE = 5e-4
BATCH_SIZE = 4  # Small batch size for CPU
NUM_EPOCHS = 12
NUM_CLASSES = 2
MODEL_NAME = "uni2-h"
SEED = 42

logger.info("=== STARTING CPU-ONLY UNI FINE-TUNING ===")
logger.info(f"Configuration:")
logger.info(f"- Device: {DEVICE} (CPU-only forced)")
logger.info(f"- Learning Rate: {LEARNING_RATE}")
logger.info(f"- Batch Size: {BATCH_SIZE}")
logger.info(f"- Epochs: {NUM_EPOCHS}")
logger.info(f"- Model: {MODEL_NAME}")

# CPU-specific optimization
torch.set_num_threads(8)  # Use 8 CPU threads
logger.info(f"Set CPU threads: {torch.get_num_threads()}")

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
logger.info(f"Random seeds set to {SEED}")

# Paths to your data
TRAIN_RESPONSE_PATH = "/cluster/projects/bhatgroup/response_to_steroid/path_slides/Response/train/patches_response_portal_tract"
TRAIN_NO_RESPONSE_PATH = "/cluster/projects/bhatgroup/response_to_steroid/path_slides/No_Response/train/patches_no_response_portal_tract"
TEST_RESPONSE_PATH = "/cluster/projects/bhatgroup/response_to_steroid/path_slides/Response/test/patches_response_portal_tract"
TEST_NO_RESPONSE_PATH = "/cluster/projects/bhatgroup/response_to_steroid/path_slides/No_Response/test/patches_no_response_portal_tract"

logger.info("Data paths configured:")
logger.info(f"- Train Response: {TRAIN_RESPONSE_PATH}")
logger.info(f"- Train No Response: {TRAIN_NO_RESPONSE_PATH}")
logger.info(f"- Test Response: {TEST_RESPONSE_PATH}")
logger.info(f"- Test No Response: {TEST_NO_RESPONSE_PATH}")

# --- 1. Dataset and DataLoader ---
logger.info("=== DATASET PREPARATION ===")

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
    start_time = time.time()
    logger.info(f"Collecting images from {folder} with label {label}")
    
    images = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append((os.path.join(folder, f), label))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Collected {len(images)} images from {folder} in {elapsed_time:.2f}s")
    return images

# Collect training images
logger.info("Collecting training data...")
train_images_data = (collect_images_and_labels(TRAIN_RESPONSE_PATH, 0) +
                     collect_images_and_labels(TRAIN_NO_RESPONSE_PATH, 1))

# Collect test images
logger.info("Collecting test data...")
test_images_data = (collect_images_and_labels(TEST_RESPONSE_PATH, 0) +
                    collect_images_and_labels(TEST_NO_RESPONSE_PATH, 1))

logger.info(f"Dataset statistics:")
logger.info(f"- Training data: {len(train_images_data)} patches")
logger.info(f"- Test data: {len(test_images_data)} patches")

# Extract paths and labels
train_img_paths = [item[0] for item in train_images_data]
train_labels = [item[1] for item in train_images_data]
test_img_paths = [item[0] for item in test_images_data]
test_labels_gt = [item[1] for item in test_images_data]

# Count labels
train_response_count = sum(1 for l in train_labels if l == 0)
train_no_response_count = sum(1 for l in train_labels if l == 1)
test_response_count = sum(1 for l in test_labels_gt if l == 0)
test_no_response_count = sum(1 for l in test_labels_gt if l == 1)

logger.info(f"Training label distribution:")
logger.info(f"- Response (0): {train_response_count}")
logger.info(f"- No Response (1): {train_no_response_count}")
logger.info(f"Test label distribution:")
logger.info(f"- Response (0): {test_response_count}")
logger.info(f"- No Response (1): {test_no_response_count}")

# Split training data into train and validation (80-20 split)
logger.info("Splitting training data into train/validation...")
train_paths, val_paths, train_lbls, val_lbls = train_test_split(
    train_img_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
)

logger.info(f"Data split results:")
logger.info(f"- Train: {len(train_paths)} samples")
logger.info(f"- Val: {len(val_paths)} samples")
logger.info(f"- Test: {len(test_img_paths)} samples")

# --- Create and save index CSV files ---
logger.info("Creating and saving index CSV files...")

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
csv_out = "/cluster/projects/bhatgroup/response_to_steroid/index_path_separated_portal_tract.csv"
df_full.to_csv(csv_out, index_label="index")
logger.info(f"Saved image-path index to {csv_out}")
logger.info(f"Index CSV shape: {df_full.shape}")

# --- 2. Load UNI Model ---
logger.info("=== LOADING UNI MODEL ===")
model_load_start = time.time()

try:
    from uni import get_encoder
    logger.info(f"Loading UNI model '{MODEL_NAME}' on CPU...")
    base_model, model_transform = get_encoder(enc_name=MODEL_NAME, device=DEVICE)
    model_load_time = time.time() - model_load_start
    logger.info(f"Successfully loaded '{MODEL_NAME}' model in {model_load_time:.2f}s")
    logger.info(f"Model transform: {model_transform}")
    
    # Force model to CPU and log memory usage
    base_model = base_model.cpu()
    gc.collect()
    logger.info("Model moved to CPU and memory cleaned")
        
except Exception as e:
    logger.error(f"Error loading UNI model: {e}")
    raise e

# Determine feature dimension
logger.info("Determining feature dimension...")
try:
    base_model.eval()
    with torch.no_grad():
        dummy_image = Image.new('RGB', (224, 224))
        dummy_input = model_transform(dummy_image).unsqueeze(0).to(DEVICE)
        features = base_model(dummy_input)
        
        logger.info(f"Raw features shape: {features.shape}")
        logger.info(f"Raw features type: {type(features)}")
        
        if isinstance(features, torch.Tensor):
            if features.ndim == 4:  # Conv feature maps [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                logger.info(f"Processed conv features to shape: {features.shape}")
            elif features.ndim == 3:  # ViT-style [B, N_patches, D]
                if features.shape[1] > 1:
                    features = features[:, 0]  # Use CLS token
                    logger.info(f"Used CLS token, shape: {features.shape}")

    feature_dim = features.shape[-1]
    logger.info(f"Detected feature dimension: {feature_dim}")
    
    # Clear dummy tensors
    del dummy_input, features
    gc.collect()
    logger.info("Dummy tensors cleared and memory cleaned")
        
except Exception as e:
    logger.error(f"Could not determine feature dimension: {e}")
    feature_dim = 1536  # Default for uni2-h
    logger.info(f"Using default feature dimension: {feature_dim}")

# --- 3. CPU-Optimized Fine-tuned Model Architecture ---
logger.info("=== BUILDING FINE-TUNED MODEL ===")

class FineTunedUNImodel(nn.Module):
    def __init__(self, base_uni_model, num_features, num_classes_out):
        super().__init__()
        self.base_model = base_uni_model
        self.dropout = nn.Dropout(0.3)
        
        # Simplified classifier for CPU efficiency
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes_out)
        )
        
        logger.info(f"Classifier architecture:")
        logger.info(f"- Input dim: {num_features}")
        logger.info(f"- Hidden dim: 128")
        logger.info(f"- Output dim: {num_classes_out}")

    def forward(self, x):
        # CPU-only forward pass (no mixed precision)
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

# Instantiate model (CPU-only)
model_to_finetune = FineTunedUNImodel(base_model, feature_dim, NUM_CLASSES)
model_to_finetune = model_to_finetune.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model_to_finetune.parameters())
trainable_params = sum(p.numel() for p in model_to_finetune.parameters() if p.requires_grad)
logger.info(f"Model parameters:")
logger.info(f"- Total parameters: {total_params:,}")
logger.info(f"- Trainable parameters: {trainable_params:,}")

# --- 4. CPU-Optimized Data Loaders ---
logger.info("=== CREATING DATA LOADERS ===")

train_dataset = CustomImageDataset(train_paths, train_lbls, transform=model_transform)
val_dataset = CustomImageDataset(val_paths, val_lbls, transform=model_transform)
test_dataset = CustomImageDataset(test_img_paths, test_labels_gt, transform=model_transform)

# CPU-optimized settings
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=0, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=False)

logger.info(f"DataLoader configuration:")
logger.info(f"- Train batches: {len(train_loader)}")
logger.info(f"- Val batches: {len(val_loader)}")
logger.info(f"- Test batches: {len(test_loader)}")
logger.info(f"- Batch size: {BATCH_SIZE}")
logger.info(f"- CPU workers: 0 (optimal for CPU)")

# --- 5. Training Setup ---
logger.info("=== TRAINING SETUP ===")

# Start with frozen backbone
frozen_params = 0
for param in model_to_finetune.base_model.parameters():
    param.requires_grad = False
    frozen_params += param.numel()

trainable_params = sum(p.numel() for p in model_to_finetune.parameters() if p.requires_grad)
logger.info(f"Initial training setup:")
logger.info(f"- Frozen backbone parameters: {frozen_params:,}")
logger.info(f"- Trainable classifier parameters: {trainable_params:,}")

# Initial optimizer for classifier only
optimizer = optim.AdamW(
    model_to_finetune.classifier.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
logger.info(f"Optimizer: AdamW with LR={LEARNING_RATE}")

# Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
logger.info(f"Loss function: CrossEntropyLoss with label_smoothing=0.1")

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=2, factor=0.5, verbose=True
)
logger.info(f"Scheduler: ReduceLROnPlateau with patience=2, factor=0.5")

# --- 6. CPU-Optimized Training Loop ---
logger.info("=== STARTING TRAINING ===")

best_val_accuracy = 0.0
best_model_path = "/cluster/projects/bhatgroup/response_to_steroid/finetuned_uni_model_best.pth"
patience_counter = 0
max_patience = 4

training_start_time = time.time()
logger.info(f"Training started at {datetime.now()}")
logger.info(f"Training for {NUM_EPOCHS} epochs on {DEVICE}")

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    logger.info(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ===")
    
    # Unfreeze backbone after 4 epochs
    if epoch == 4:
        logger.info("Unfreezing backbone for full fine-tuning...")
        
        unfrozen_params = 0
        for param in model_to_finetune.base_model.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
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
        
        logger.info(f"Backbone unfrozen: {unfrozen_params:,} parameters")
        logger.info(f"Differential LRs - Backbone: {backbone_lr:.2e}, Classifier: {classifier_lr:.2e}")
    
    # Training phase
    logger.info(f"Starting training phase for epoch {epoch+1}")
    model_to_finetune.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # CPU tensor transfer
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        
        # CPU forward pass (no mixed precision)
        outputs = model_to_finetune(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_to_finetune.parameters(), max_norm=1.0)
        optimizer.step()

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
        
        # Log every 20 batches and clean memory
        if batch_idx % 20 == 0:
            batch_time = time.time() - batch_start_time
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}, "
                       f"acc={100.*correct_train/total_train:.2f}%, time={batch_time:.2f}s")
            gc.collect()

    train_time = time.time() - train_start_time
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc_train = 100. * correct_train / len(train_loader.dataset)
    
    logger.info(f"Training phase completed in {train_time:.2f}s")
    logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%")

    # Validation phase
    logger.info(f"Starting validation phase for epoch {epoch+1}")
    model_to_finetune.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # CPU forward pass
            outputs = model_to_finetune(images)
            loss = criterion(outputs, labels)
                
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_time = time.time() - val_start_time
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_acc_val = 100. * correct_val / len(val_loader.dataset)
    
    logger.info(f"Validation phase completed in {val_time:.2f}s")
    logger.info(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_acc_val:.2f}%")
    
    current_lr = optimizer.param_groups[0]['lr']
    epoch_total_time = time.time() - epoch_start_time
    
    logger.info(f"EPOCH {epoch+1} SUMMARY:")
    logger.info(f"- Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%")
    logger.info(f"- Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_acc_val:.2f}%")
    logger.info(f"- Learning Rate: {current_lr:.1e}")
    logger.info(f"- Total Epoch Time: {epoch_total_time:.2f}s")

    # Step scheduler
    scheduler.step(epoch_val_loss)

    # Save best model with early stopping
    if epoch_acc_val > best_val_accuracy:
        best_val_accuracy = epoch_acc_val
        patience_counter = 0
        
        save_start_time = time.time()
        torch.save({
            'model_state_dict': model_to_finetune.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_accuracy': best_val_accuracy,
            'feature_dim': feature_dim,
            'num_classes': NUM_CLASSES,
            'learning_rate': LEARNING_RATE,
            'device': str(DEVICE),
            'training_time': time.time() - training_start_time
        }, best_model_path)
        save_time = time.time() - save_start_time
        
        logger.info(f"✓ NEW BEST MODEL SAVED (Val Acc: {best_val_accuracy:.2f}%)")
        logger.info(f"Model saved to {best_model_path} in {save_time:.2f}s")
    else:
        patience_counter += 1
        logger.info(f"No improvement. Patience: {patience_counter}/{max_patience}")
        
        if patience_counter >= max_patience and epoch > 6:
            logger.info(f"EARLY STOPPING triggered after {patience_counter} epochs without improvement")
            break
    
    # Memory cleanup after each epoch
    gc.collect()

total_training_time = time.time() - training_start_time
logger.info(f"\n=== TRAINING COMPLETED ===")
logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
logger.info(f"Training finished at {datetime.now()}")

# --- 7. CPU-Optimized Testing ---
logger.info("=== TESTING FINE-TUNED MODEL ===")
test_start_time = time.time()

logger.info("Loading best model for testing...")
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model_to_finetune.load_state_dict(checkpoint['model_state_dict'])
model_to_finetune.eval()

logger.info(f"Best model loaded from epoch {checkpoint['epoch']+1}")
logger.info(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")

# Memory cleanup before testing
gc.collect()

logger.info("Starting testing phase...")
test_loss = 0.0
correct_test = 0
total_test = 0
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # CPU forward pass
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
        
        # Log and clean memory every 10 batches
        if batch_idx % 10 == 0:
            logger.info(f"Test batch {batch_idx}/{len(test_loader)} completed")
            gc.collect()

test_time = time.time() - test_start_time
test_accuracy = 100. * correct_test / total_test
test_loss_avg = test_loss / len(test_loader.dataset)

logger.info(f"=== TESTING COMPLETED ===")
logger.info(f"Test time: {test_time:.2f}s")
logger.info(f"Fine-tuned Model Test Accuracy: {test_accuracy:.2f}%")
logger.info(f"Fine-tuned Model Test Loss: {test_loss_avg:.4f}")
logger.info(f"Total test samples: {total_test}")
logger.info(f"Correct predictions: {correct_test}")

# --- 8. Extract Features (CPU-Optimized) ---
logger.info("=== EXTRACTING FEATURES FOR EVALUATION ===")
feature_extraction_start = time.time()

# Free up model memory temporarily
del model_to_finetune
gc.collect()
logger.info("Fine-tuned model removed from memory")

from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

# Create smaller batch size loaders for feature extraction
feature_batch_size = 2  # Very small for CPU
logger.info(f"Creating feature extraction dataloaders with batch_size={feature_batch_size}")

train_dataloader_for_features = DataLoader(train_dataset, batch_size=feature_batch_size, 
                                         shuffle=False, num_workers=0, pin_memory=False)
test_dataloader_for_features = DataLoader(test_dataset, batch_size=feature_batch_size, 
                                        shuffle=False, num_workers=0, pin_memory=False)

logger.info("Extracting training features...")
train_feature_start = time.time()
train_features = extract_patch_features_from_dataloader(base_model, train_dataloader_for_features)
train_feature_time = time.time() - train_feature_start
logger.info(f"Training features extracted in {train_feature_time:.2f}s")
gc.collect()
    
logger.info("Extracting test features...")
test_feature_start = time.time()
test_features = extract_patch_features_from_dataloader(base_model, test_dataloader_for_features)
test_feature_time = time.time() - test_feature_start
logger.info(f"Test features extracted in {test_feature_time:.2f}s")
gc.collect()

# Convert to tensors
train_feats = torch.Tensor(train_features['embeddings'])
train_labels_tensor = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels_tensor = torch.Tensor(test_features['labels']).type(torch.long)

total_feature_time = time.time() - feature_extraction_start
logger.info(f"=== FEATURE EXTRACTION COMPLETED ===")
logger.info(f"Total feature extraction time: {total_feature_time:.2f}s")
logger.info(f"Train features shape: {train_feats.shape}")
logger.info(f"Test features shape: {test_feats.shape}")

# Move base model to CPU to free memory
base_model.cpu()
gc.collect()
logger.info("Base model moved to CPU and memory cleaned")

# --- 9. Evaluation Functions ---
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
    eval_start_time = time.time()
    logger.info(f"Starting {prefix} classifier evaluation...")
    
    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_feats.cpu().numpy()
    y_test = test_labels.cpu().numpy()
    
    scaler = None
    if scale_features:
        scaler_start = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        scaler_time = time.time() - scaler_start
        logger.info(f"{prefix} feature scaling completed in {scaler_time:.2f}s")
    
    fit_start_time = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - fit_start_time
    logger.info(f"{prefix} classifier fitted in {fit_time:.2f}s")
    
    pred_start_time = time.time()
    preds_all = classifier.predict(X_test)
    pred_time = time.time() - pred_start_time
    logger.info(f"{prefix} predictions completed in {pred_time:.2f}s")
    
    probs_all = None
    if hasattr(classifier, "predict_proba"):
        try:
            prob_start_time = time.time()
            probs_all = classifier.predict_proba(X_test)
            prob_time = time.time() - prob_start_time
            logger.info(f"{prefix} probabilities computed in {prob_time:.2f}s")
        except Exception as e:
            logger.warning(f"{prefix} probability computation failed: {e}")
    
    metrics = get_eval_metrics(y_test, preds_all, probs_all, prefix=prefix)
    
    total_eval_time = time.time() - eval_start_time
    logger.info(f"{prefix} evaluation completed in {total_eval_time:.2f}s")
    
    return metrics, {
        "preds_all": preds_all,
        "probs_all": probs_all,
        "targets_all": y_test,
        "classifier": classifier,
        "scaler": scaler
    }

# --- 10. Baseline Method Evaluations ---
logger.info("=== BASELINE METHOD EVALUATIONS ===")
baseline_start_time = time.time()

# Linear Probe
logger.info("=== LINEAR PROBE EVALUATION ===")
try:
    from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
    linprobe_start = time.time()
    linprobe_metrics, linprobe_dump = eval_linear_probe(
        train_feats=train_feats, train_labels=train_labels_tensor,
        valid_feats=None, valid_labels=None,
        test_feats=test_feats, test_labels=test_labels_tensor,
        max_iter=1000, verbose=True
    )
    linprobe_time = time.time() - linprobe_start
    logger.info(f"Linear Probe completed in {linprobe_time:.2f}s")
    logger.info(f"Linear Probe Accuracy: {linprobe_metrics.get('lin_acc', 0):.4f}")
except Exception as e:
    logger.error(f"Linear Probe evaluation failed: {e}")
    linprobe_metrics = {"lin_acc": 0}

# KNN (UNI implementation)
logger.info("=== KNN (UNI) EVALUATION ===")
try:
    from uni.downstream.eval_patch_features.fewshot import eval_knn
    knn_start = time.time()
    knn_metrics, knn_dump, proto_metrics, proto_dump = eval_knn(
        train_feats=train_feats, train_labels=train_labels_tensor,
        test_feats=test_feats, test_labels=test_labels_tensor,
        center_feats=True, normalize_feats=True, n_neighbors=5
    )
    knn_time = time.time() - knn_start
    logger.info(f"KNN (UNI) completed in {knn_time:.2f}s")
    logger.info(f"KNN (UNI) Accuracy: {knn_metrics.get('knn5_acc', 0):.4f}")
except Exception as e:
    logger.error(f"KNN (UNI) evaluation failed: {e}")
    knn_metrics = {"knn5_acc": 0}

# SVM
logger.info("=== SVM EVALUATION ===")
svm_classifier = SVC(
    C=0.1, kernel='rbf', gamma='scale', 
    class_weight='balanced', probability=True, random_state=SEED
)
svm_metrics, svm_dump = eval_sklearn_classifier(
    classifier=svm_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="svm_", scale_features=True
)
logger.info(f"SVM Accuracy: {svm_metrics.get('svm_acc', 0):.4f}")

# Random Forest
logger.info("=== RANDOM FOREST EVALUATION ===")
rf_classifier = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=3,
    min_samples_leaf=2, class_weight='balanced', 
    random_state=SEED, n_jobs=-1
)
rf_metrics, rf_dump = eval_sklearn_classifier(
    classifier=rf_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="rf_", scale_features=False
)
logger.info(f"Random Forest Accuracy: {rf_metrics.get('rf_acc', 0):.4f}")

# Gradient Boosting
logger.info("=== GRADIENT BOOSTING EVALUATION ===")
gb_classifier = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=SEED
)
gb_metrics, gb_dump = eval_sklearn_classifier(
    classifier=gb_classifier,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="gb_", scale_features=False
)
logger.info(f"Gradient Boosting Accuracy: {gb_metrics.get('gb_acc', 0):.4f}")

# KNN (sklearn)
logger.info("=== KNN (SKLEARN) EVALUATION ===")
knn_sklearn = KNeighborsClassifier(
    n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
)
knn_sklearn_metrics, knn_sklearn_dump = eval_sklearn_classifier(
    classifier=knn_sklearn,
    train_feats=train_feats, train_labels=train_labels_tensor,
    test_feats=test_feats, test_labels=test_labels_tensor,
    prefix="knn_sklearn_", scale_features=True
)
logger.info(f"KNN (sklearn) Accuracy: {knn_sklearn_metrics.get('knn_sklearn_acc', 0):.4f}")

baseline_time = time.time() - baseline_start_time
logger.info(f"=== ALL BASELINE EVALUATIONS COMPLETED ===")
logger.info(f"Total baseline evaluation time: {baseline_time:.2f}s")

# --- 11. Find Best Model ---
logger.info("=== FINDING BEST MODEL ===")

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

logger.info(f"=== MODEL COMPARISON RESULTS ===")
for i, (model_name, accuracy) in enumerate(sorted_results):
    rank_symbol = "🏆" if i == 0 else f"{i+1}."
    logger.info(f"{rank_symbol} {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

logger.info(f"Best performing model: {best_model_name} with {best_accuracy:.4f} accuracy")

# Get best model predictions and probabilities
logger.info("Retrieving best model predictions and probabilities...")
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

logger.info(f"Best model predictions shape: {best_predictions.shape}")
if best_probabilities is not None:
    logger.info(f"Best model probabilities shape: {best_probabilities.shape}")

# --- 12. Create Test Images DataFrame ---
logger.info("=== CREATING TEST IMAGES DATAFRAME ===")
test_df_start = time.time()

test_imgs_df = pd.DataFrame(test_dataset.data, columns=['path', 'label'])
test_df_path = '/cluster/projects/bhatgroup/response_to_steroid/test_imgs_df_portal_tract_separated.csv'
test_imgs_df.to_csv(test_df_path, index=False)

test_df_time = time.time() - test_df_start
logger.info(f"Test images DataFrame created and saved in {test_df_time:.2f}s")
logger.info(f"Test DataFrame shape: {test_imgs_df.shape}")
logger.info(f"Saved to: {test_df_path}")

# --- 13. Top-k Patch Retrieval using Best Model ---
logger.info(f"=== TOP-K PATCH RETRIEVAL (BEST MODEL: {best_model_name}) ===")
topk_start_time = time.time()

# Calculate confidence scores for ranking
logger.info("Calculating confidence scores for ranking...")
if best_probabilities is not None:
    if best_probabilities.ndim == 2 and best_probabilities.shape[1] == 2:
        confidence_response = best_probabilities[:, 0]
        confidence_no_response = best_probabilities[:, 1]
        logger.info("Using model probabilities for confidence scores")
    else:
        confidence_response = (1 - best_predictions) + np.random.rand(len(best_predictions)) * 0.1
        confidence_no_response = best_predictions + np.random.rand(len(best_predictions)) * 0.1
        logger.info("Using prediction-based confidence (1D probabilities)")
else:
    logger.info("No probabilities available, using prediction-based confidence")
    confidence_response = (1 - best_predictions) + np.random.rand(len(best_predictions)) * 0.1
    confidence_no_response = best_predictions + np.random.rand(len(best_predictions)) * 0.1

# Get all indices sorted by confidence for each class
response_indices = np.argsort(confidence_response)[::-1]
no_response_indices = np.argsort(confidence_no_response)[::-1]

logger.info(f"Response indices range: {response_indices[:5]} ... {response_indices[-5:]}")
logger.info(f"No response indices range: {no_response_indices[:5]} ... {no_response_indices[-5:]}")

# Try to import concat_images
try:
    from uni.downstream.utils import concat_images
    logger.info("concat_images imported successfully")
except ImportError:
    logger.warning("concat_images not available, will save indices only")
    concat_images = None

# Function to save top-k patches
def save_topk_patches(indices, class_name, k_values, confidence_scores):
    logger.info(f"Saving top-k patches for {class_name}...")
    
    # Save all top-k indices to CSV
    df_indices = pd.DataFrame({
        'index': indices[:max(k_values)],
        'confidence': confidence_scores[indices[:max(k_values)]]
    })
    csv_path = f"/cluster/projects/bhatgroup/response_to_steroid/topk_{class_name}_indices_portal_tract_separated.csv"
    df_indices.to_csv(csv_path, index=False)
    logger.info(f"Saved top-k indices for {class_name} to {csv_path}")
    
    # Generate concatenated images for different k values
    if concat_images is not None:
        for k in k_values:
            if len(indices) >= k:
                image_start_time = time.time()
                topk_indices = indices[:k]
                images = [Image.open(test_imgs_df['path'].iloc[idx]) for idx in topk_indices]
                concat_img = concat_images(images, gap=5)
                save_path = f"/cluster/projects/bhatgroup/response_to_steroid/Top{k}_{class_name}_portal_tract_separated_{best_model_name.replace(' ', '_')}.png"
                concat_img.save(save_path)
                image_time = time.time() - image_start_time
                logger.info(f"Saved concatenated top {k} {class_name} image to {save_path} in {image_time:.2f}s")
                
                # Also save individual top-k indices CSV
                df_topk = pd.DataFrame({
                    'index': topk_indices,
                    'confidence': confidence_scores[topk_indices]
                })
                csv_topk_path = f"/cluster/projects/bhatgroup/response_to_steroid/top{k}_{class_name}_indices_portal_tract_separated.csv"
                df_topk.to_csv(csv_topk_path, index=False)
                logger.info(f"Saved top-{k} individual indices for {class_name}")

# Save top-k patches for both classes
k_values = [5, 10, 15, 20, 25]
logger.info(f"Processing k values: {k_values}")

logger.info("=== TOP-K RESPONSE PATCHES ===")
save_topk_patches(response_indices, "response", k_values, confidence_response)

logger.info("=== TOP-K NO RESPONSE PATCHES ===")
save_topk_patches(no_response_indices, "no_response", k_values, confidence_no_response)

topk_time = time.time() - topk_start_time
logger.info(f"Top-k patch retrieval completed in {topk_time:.2f}s")

# --- 14. Slide-level Evaluation using Best Model ---
logger.info(f"=== SLIDE-LEVEL EVALUATION (BEST MODEL: {best_model_name}) ===")
slide_eval_start = time.time()

# Dictionary to collect patch predictions for each slide
slide_preds = defaultdict(list)
slide_actual = {}

logger.info("Collecting patch predictions for each slide...")
processed_count = 0
for (img_path, actual_label), pred_label in zip(test_dataset.data, best_predictions):
    basename = os.path.basename(img_path)
    parts = basename.split('-')
    if len(parts) < 2:
        logger.warning(f"Filename format unexpected: {basename}")
        continue
    
    slide_id = parts[1]
    slide_preds[slide_id].append(pred_label)
    
    if slide_id not in slide_actual:
        slide_actual[slide_id] = actual_label
    else:
        if slide_actual[slide_id] != actual_label:
            logger.warning(f"Inconsistent actual labels for slide {slide_id}")
    
    processed_count += 1

logger.info(f"Processed {processed_count} patches across {len(slide_preds)} slides")

# Compute majority vote prediction for each slide
logger.info("Computing majority vote predictions for each slide...")
slide_pred_majority = {}
slide_pred_confidence = {}
for slide, preds in slide_preds.items():
    avg_pred = np.mean(preds)
    majority_label = 0 if avg_pred < 0.5 else 1
    slide_pred_majority[slide] = majority_label
    slide_pred_confidence[slide] = abs(avg_pred - 0.5) * 2

logger.info(f"Slide predictions computed for {len(slide_pred_majority)} slides")

# Calculate slide-level accuracy
logger.info("Calculating slide-level metrics...")
y_true_slides = []
y_pred_slides = []
slide_confidences = []
for slide in slide_actual:
    y_true_slides.append(slide_actual[slide])
    y_pred_slides.append(slide_pred_majority.get(slide, 0))
    slide_confidences.append(slide_pred_confidence.get(slide, 0))

slide_accuracy = np.mean(np.array(y_true_slides) == np.array(y_pred_slides))
slide_bacc = balanced_accuracy_score(y_true_slides, y_pred_slides)

logger.info(f"=== SLIDE-LEVEL RESULTS ===")
logger.info(f"Total slides evaluated: {len(y_true_slides)}")
logger.info(f"Slide-level Accuracy: {slide_accuracy:.4f} ({slide_accuracy*100:.2f}%)")
logger.info(f"Slide-level Balanced Accuracy: {slide_bacc:.4f} ({slide_bacc*100:.2f}%)")

# Compute and log the confusion matrix
cm = confusion_matrix(y_true_slides, y_pred_slides)
logger.info("Slide-level Confusion Matrix (rows: actual, columns: predicted):")
logger.info(f"\n{cm}")

# Additional slide-level metrics
slide_metrics = get_eval_metrics(y_true_slides, y_pred_slides, prefix="slide_")
logger.info("Detailed Slide-level Metrics:")
for k, v in slide_metrics.items():
    if "report" not in k:
        logger.info(f"  {k}: {v:.4f}")

# Save slide-level results
logger.info("Saving slide-level results...")
slide_results_df = pd.DataFrame({
    'slide_id': list(slide_actual.keys()),
    'actual_label': [slide_actual[sid] for sid in slide_actual.keys()],
    'predicted_label': [slide_pred_majority.get(sid, 0) for sid in slide_actual.keys()],
    'confidence': [slide_pred_confidence.get(sid, 0) for sid in slide_actual.keys()],
    'num_patches': [len(slide_preds[sid]) for sid in slide_actual.keys()],
    'correct_prediction': [slide_actual[sid] == slide_pred_majority.get(sid, 0) for sid in slide_actual.keys()]
})
slide_results_path = f"/cluster/projects/bhatgroup/response_to_steroid/slide_level_results_portal_tract_separated_{best_model_name.replace(' ', '_')}.csv"
slide_results_df.to_csv(slide_results_path, index=False)

slide_eval_time = time.time() - slide_eval_start
logger.info(f"Slide-level evaluation completed in {slide_eval_time:.2f}s")
logger.info(f"Slide-level results saved to {slide_results_path}")

# --- 15. Save All Results ---
logger.info("=== SAVING ALL RESULTS ===")
save_start_time = time.time()

# Save fine-tuned model predictions
logger.info("Saving fine-tuned model results...")
finetuned_results = {
    'predictions': all_predictions,
    'labels': all_labels,
    'probabilities': all_probabilities,
    'test_accuracy': test_accuracy,
    'test_loss': test_loss_avg,
    'best_val_accuracy': best_val_accuracy,
    'learning_rate_used': LEARNING_RATE,
    'epochs_trained': checkpoint['epoch'] + 1,
    'device_used': str(DEVICE),
    'total_training_time': total_training_time,
    'total_test_time': test_time
}

finetuned_path = '/cluster/projects/bhatgroup/response_to_steroid/finetuned_uni_test_results.pkl'
with open(finetuned_path, 'wb') as f:
    pickle.dump(finetuned_results, f)
logger.info(f"Fine-tuned results saved to {finetuned_path}")

# Save all results summary
logger.info("Saving comprehensive results summary...")
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
        'dataset_size': len(train_paths),
        'device': str(DEVICE),
        'cpu_threads': torch.get_num_threads(),
        'model_name': MODEL_NAME,
        'seed': SEED
    },
    'timing': {
        'total_training_time': total_training_time,
        'total_test_time': test_time,
        'feature_extraction_time': total_feature_time,
        'baseline_evaluation_time': baseline_time,
        'slide_evaluation_time': slide_eval_time,
        'topk_retrieval_time': topk_time
    }
}

all_results_path = '/cluster/projects/bhatgroup/response_to_steroid/all_evaluation_results.pkl'
with open(all_results_path, 'wb') as f:
    pickle.dump(all_results, f)
logger.info(f"All results saved to {all_results_path}")

# Create results DataFrame
logger.info("Creating model comparison CSV...")
results_df = pd.DataFrame([
    {'Model': name, 'Accuracy': acc, 'Percentage': f"{acc*100:.2f}%"}
    for name, acc in sorted_results
])
results_csv_path = '/cluster/projects/bhatgroup/response_to_steroid/model_comparison_results.csv'
results_df.to_csv(results_csv_path, index=False)
logger.info(f"Model comparison results saved to {results_csv_path}")

save_time = time.time() - save_start_time
logger.info(f"All results saved in {save_time:.2f}s")

# --- Final Summary ---
total_execution_time = time.time() - training_start_time
logger.info("\n" + "="*60)
logger.info("=== FINAL EXECUTION SUMMARY ===")
logger.info("="*60)
logger.info(f"🎯 Execution completed successfully on {DEVICE}")
logger.info(f"📅 Finished at: {datetime.now()}")
logger.info(f"⏱️  Total execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f} minutes)")
logger.info("")
logger.info("📊 PERFORMANCE RESULTS:")
logger.info(f"🏆 Best model: {best_model_name} ({best_accuracy*100:.2f}%)")
logger.info(f"📈 Fine-tuned UNI accuracy: {test_accuracy:.2f}%")
logger.info(f"🔍 Slide-level accuracy: {slide_accuracy*100:.2f}%")
logger.info("")
logger.info("⏱️  TIMING BREAKDOWN:")
logger.info(f"- Training: {total_training_time:.2f}s ({total_training_time/60:.2f}m)")
logger.info(f"- Testing: {test_time:.2f}s")
logger.info(f"- Feature extraction: {total_feature_time:.2f}s ({total_feature_time/60:.2f}m)")
logger.info(f"- Baseline evaluation: {baseline_time:.2f}s ({baseline_time/60:.2f}m)")
logger.info(f"- Slide evaluation: {slide_eval_time:.2f}s")
logger.info(f"- Top-k retrieval: {topk_time:.2f}s")
logger.info("")
logger.info("💾 FILES SAVED:")
logger.info("- index_path_separated_portal_tract.csv")
logger.info("- test_imgs_df_portal_tract_separated.csv")
logger.info("- finetuned_uni_test_results.pkl")
logger.info("- all_evaluation_results.pkl")
logger.info("- model_comparison_results.csv")
logger.info(f"- slide_level_results_portal_tract_separated_{best_model_name.replace(' ', '_')}.csv")
logger.info(f"- Log file: /cluster/projects/bhatgroup/response_to_steroid/finetune_uni_cpu.log")
logger.info("")
logger.info("✅ CPU-OPTIMIZED FINE-TUNING COMPLETED SUCCESSFULLY!")
logger.info("="*60)