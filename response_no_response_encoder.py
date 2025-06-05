import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score,
    cohen_kappa_score, classification_report, confusion_matrix, f1_score
)
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from utils.my_utils import extract_features, log_output, get_eval_metrics, eval_sklearn_classifier

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
import umap.umap_ as umap
import tqdm

# Set seed for reproducibility
seed_value = 42
SEED=42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
response_path = "patches_mix_no_response_portal_tract"
no_response_path = "patches_mix_no_response_portal_tract"

# Collect all image file paths and their respective labels
seed_value = 42               # reproducible shuffling
batch_size = 16

# Shuffle and split into train and test sets
def collect_images(folder, label):
    return [(os.path.join(folder, f), label)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

response_images = collect_images(response_path, 0)
no_response_images = collect_images(no_response_path, 1)

def make_df(image_list):
    """Convert list[(path, label)] → DataFrame."""
    img_paths = [p for p, _ in image_list]
    labels    = [l for _, l in image_list]
    classes   = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({
        "image_path": img_paths,
        "label":      labels,          # 0 or 1
        "class_name": classes          # human‑readable
    })

df_response = make_df(response_images)
df_response.to_csv('response_images.csv', index=False)
log_output(f"Response images saved to response_images.csv with {len(df_response)} entries.")
df_no_response = make_df(no_response_images)
df_no_response.to_csv('no_response_images.csv', index=False)
log_output(f"No Response images saved to no_response_images.csv with {len(df_no_response)} entries.")


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data      = data          # list of (filepath, label)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load encoder & transform
from uni import get_encoder
model, transform = get_encoder(enc_name="uni2-h", device=device)

# Build datasets/dataloaders
response_dataset = CustomImageDataset(response_images, transform=transform)
no_response_dataset = CustomImageDataset(no_response_images, transform=transform)

train_dataloader = torch.utils.data.DataLoader(response_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_dataloader  = torch.utils.data.DataLoader(no_response_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4)

# # Extract features
# response_features = extract_features(model, train_dataloader)
# no_response_features = extract_features(model, test_dataloader)

# # Save response_features and no_response_features to files
# with open('response_features_portal_tract.pkl', 'wb') as f:
#     pickle.dump(response_features, f)

# with open('no_response_features_portal_tract.pkl', 'wb') as f:
#     pickle.dump(no_response_features, f)

# Load features from files
with open('response_features_portal_tract.pkl', 'rb') as f:
    response_features = pickle.load(f)
with open('no_response_features_portal_tract.pkl', 'rb') as f:
    no_response_features = pickle.load(f)

log_output("Clinical Data Processing ...")

# ──────────────────────────── 9. Load Clinical Data ─────────────────────────

# Load clinical data
with open('df_cleaned_normalized.pkl', 'rb') as f:
    df_cleaned = pickle.load(f)

X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"])
y_clinical = df_cleaned['RAI Classification Biopsy #2']
patient_ids = df_cleaned['patient_id']

import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score,
    cohen_kappa_score, classification_report, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

from utils.my_utils import log_output, get_eval_metrics, eval_sklearn_classifier

# Set up logging
log_file = "k_fold_multimodal_fusion.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def calculate_feature_weights():
    """Calculate weights based on feature dimensions"""
    img_dims = 1536  # Vision Transformer feature dimension
    clinical_dims = 13  # Clinical feature dimension
    total_dims = img_dims + clinical_dims
    img_weight = img_dims / total_dims
    clinical_weight = clinical_dims / total_dims
    
    log_output(f"Dimension-based weights - Image: {img_weight:.4f}, Clinical: {clinical_weight:.4f}")
    return img_weight, clinical_weight

def train_eval_sklearn_model(model, X_train, y_train, X_test, y_test, model_name, scale_features=False):
    """Train and evaluate a scikit-learn model"""
    log_output(f"Training {model_name}...")
    
    # Scale features if needed
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    if y_prob is not None and len(np.unique(y_test)) > 1:
        if y_prob.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multi-class classification
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc = None
    
    log_output(f"{model_name} - Accuracy: {accuracy:.4f}, Balanced Acc: {balanced_acc:.4f}")
    if auc is not None:
        log_output(f"{model_name} - AUC: {auc:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def get_slide_level_predictions(test_dataset_items, predictions, probabilities=None):
    """Aggregate patch-level predictions to slide-level predictions"""
    slide_preds = defaultdict(list)
    slide_probs = defaultdict(list)
    slide_actual = {}
    
    for (img_path, actual_label), pred_label in zip(test_dataset_items, predictions):
        basename = os.path.basename(img_path)
        parts = basename.split('-')
        if len(parts) < 2:
            log_output(f"Filename format unexpected: {basename}")
            continue
        
        # Extract slide_id (patient_id)
        slide_id = parts[1]
        
        # Store prediction for this slide
        slide_preds[slide_id].append(pred_label)
        
        # Store probability if available
        if probabilities is not None:
            if isinstance(probabilities, np.ndarray):
                if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    # Binary classification probabilities for positive class
                    prob = probabilities[_][1]
                    slide_probs[slide_id].append(prob)
                else:
                    # Use prediction with some uncertainty
                    slide_probs[slide_id].append(float(pred_label))
            else:
                slide_probs[slide_id].append(float(pred_label))
        
        # Set the actual label for the slide
        if slide_id not in slide_actual:
            slide_actual[slide_id] = actual_label
        else:
            if slide_actual[slide_id] != actual_label:
                log_output(f"Warning: inconsistent labels for slide {slide_id}")
    
    # Compute majority vote prediction for each slide
    slide_pred_majority = {}
    slide_prob_avg = {}
    for slide, preds in slide_preds.items():
        # Majority vote: if average is less than 0.5 then label 0; otherwise label 1
        avg_pred = np.mean(preds)
        majority_label = 0 if avg_pred < 0.5 else 1
        slide_pred_majority[slide] = majority_label
    
    # Compute average probability for each slide
    for slide, probs in slide_probs.items():
        slide_prob_avg[slide] = np.mean(probs)
    
    # Prepare result in format suitable for evaluation
    slide_ids = list(slide_actual.keys())
    y_true_slides = [slide_actual[sid] for sid in slide_ids]
    y_pred_slides = [slide_pred_majority[sid] for sid in slide_ids]
    y_prob_slides = [slide_prob_avg.get(sid, 0.5) for sid in slide_ids] if slide_probs else None
    
    return slide_ids, y_true_slides, y_pred_slides, y_prob_slides

def main():
    log_output("=== Starting 5-Fold Cross-Validation for Multimodal Fusion ===")
    
    # ------ 1. Load Clinical Data ------
    log_output("\n=== Loading Clinical Data ===")
    with open('df_cleaned_normalized.pkl', 'rb') as f:
        df_cleaned = pickle.load(f)
    
    X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"]).values
    y_clinical = df_cleaned['RAI Classification Biopsy #2'].values
    patient_ids_clinical = df_cleaned['patient_id'].values
    
    log_output(f"Clinical data shape: {X_clinical.shape}")
    log_output(f"Number of patients in clinical data: {len(patient_ids_clinical)}")
    
    # ------ 2. Load Pathology Data ------
    log_output("\n=== Loading Pathology Data ===")
    # Load train_features and test_features from files
    with open('train_features_mix_portal_tract.pkl', 'rb') as f:
        loaded_train_features = pickle.load(f)
    
    with open('test_features_mix_portal_tract.pkl', 'rb') as f:
        loaded_test_features = pickle.load(f)
    
    # Combine train and test features for cross-validation
    X_pathology = np.vstack([
        loaded_train_features['embeddings'],
        loaded_test_features['embeddings']
    ])
    
    y_pathology = np.concatenate([
        loaded_train_features['labels'],
        loaded_test_features['labels']
    ])
    
    # Combine paths
    all_paths = loaded_train_features['paths'] + loaded_test_features['paths']
    
    # Extract patient IDs from paths
    patient_ids_pathology = []
    for path in all_paths:
        basename = os.path.basename(path)
        parts = basename.split('-')
        if len(parts) >= 2:
            patient_ids_pathology.append(parts[1])
        else:
            patient_ids_pathology.append("unknown")
    
    # Create a dataset structure for pathology data
    pathology_dataset_items = list(zip(all_paths, y_pathology))
    
    log_output(f"Pathology data shape: {X_pathology.shape}")
    log_output(f"Number of patches: {len(pathology_dataset_items)}")
    
    # ------ 3. Set Up K-Fold Cross-Validation ------
    n_folds = 5
    kf_clinical = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays to store results
    fold_results = []
    
    # Calculate feature dimension weights once
    img_weight, clinical_weight = calculate_feature_weights()
    
    # ------ 4. Perform Cross-Validation ------
    for fold, (train_idx_clinical, test_idx_clinical) in enumerate(kf_clinical.split(X_clinical)):
        log_output(f"\n=== Fold {fold+1}/{n_folds} ===")
        
        # --- 4.1. Split Clinical Data ---
        X_clinical_train = X_clinical[train_idx_clinical]
        y_clinical_train = y_clinical[train_idx_clinical]
        X_clinical_test = X_clinical[test_idx_clinical]
        y_clinical_test = y_clinical[test_idx_clinical]
        patient_ids_clinical_test = patient_ids_clinical[test_idx_clinical]
        
        # --- 4.2. Split Pathology Data Based on Patient IDs ---
        log_output("Splitting pathology data based on patient IDs...")
        
        # Get unique patient IDs in test set
        test_patients = set(patient_ids_clinical_test)
        
        # Separate pathology data into train/test based on patient IDs
        train_items = []
        test_items = []
        train_indices = []
        test_indices = []
        
        for i, (path, label) in enumerate(pathology_dataset_items):
            basename = os.path.basename(path)
            parts = basename.split('-')
            if len(parts) >= 2:
                patient_id = parts[1]
                if patient_id in test_patients:
                    test_items.append((path, label))
                    test_indices.append(i)
                else:
                    train_items.append((path, label))
                    train_indices.append(i)
        
        X_pathology_train = X_pathology[train_indices]
        y_pathology_train = y_pathology[train_indices]
        X_pathology_test = X_pathology[test_indices]
        y_pathology_test = y_pathology[test_indices]
        
        log_output(f"Pathology train set: {len(train_items)} patches")
        log_output(f"Pathology test set: {len(test_items)} patches")
        
        # --- 4.3. Train Clinical Models ---
        log_output("\n--- Training Clinical Models ---")
        
        # Initialize clinical models
        clinical_models = {
            'lr': sk_LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42),
            'svm': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        }
        
        # Train and evaluate each clinical model
        clinical_results = {}
        for name, model in clinical_models.items():
            scale = True if name in ['lr', 'svm'] else False
            clinical_results[name] = train_eval_sklearn_model(
                model, X_clinical_train, y_clinical_train, 
                X_clinical_test, y_clinical_test, 
                f"Clinical {name.upper()}", scale_features=scale
            )
        
        # --- 4.4. Train Pathology Models ---
        log_output("\n--- Training Pathology Models ---")
        
        # Initialize pathology models
        pathology_models = {
            'lr': sk_LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42),
            'svm': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        }
        
        # Train and evaluate each pathology model
        pathology_results = {}
        for name, model in pathology_models.items():
            scale = True if name in ['lr', 'svm'] else False
            # Use eval_sklearn_classifier from my_utils for pathology models
            metrics, dump = eval_sklearn_classifier(
                model, torch.tensor(X_pathology_train), torch.tensor(y_pathology_train),
                torch.tensor(X_pathology_test), torch.tensor(y_pathology_test),
                prefix=f"pathology_{name}_", scale_features=scale
            )
            
            # Convert to format compatible with clinical results
            pathology_results[name] = {
                'model': model,
                'scaler': dump.get('scaler'),
                'accuracy': metrics.get(f"pathology_{name}_acc", 0),
                'balanced_accuracy': metrics.get(f"pathology_{name}_bacc", 0),
                'auc': metrics.get(f"pathology_{name}_auroc", None),
                'predictions': dump['preds_all'],
                'probabilities': dump['probs_all']
            }
            
            log_output(f"Pathology {name.upper()} - Accuracy: {pathology_results[name]['accuracy']:.4f}, "
                      f"Balanced Acc: {pathology_results[name]['balanced_accuracy']:.4f}")
            
        # --- 4.5. Aggregate Pathology Predictions at Slide Level ---
        log_output("\n--- Aggregating Pathology Predictions at Slide Level ---")
        
        # Choose best pathology model based on accuracy
        best_pathology_model = max(pathology_results.keys(), 
                                 key=lambda k: pathology_results[k]['accuracy'])
        
        log_output(f"Best pathology model: {best_pathology_model.upper()} "
                 f"(Acc: {pathology_results[best_pathology_model]['accuracy']:.4f})")
        
        # Get slide-level predictions from best pathology model
        slide_ids, y_true_slides, y_pred_slides, y_prob_slides = get_slide_level_predictions(
            test_items,
            pathology_results[best_pathology_model]['predictions'],
            pathology_results[best_pathology_model]['probabilities']
        )
        
        # Evaluate slide-level accuracy
        slide_accuracy = accuracy_score(y_true_slides, y_pred_slides)
        slide_bacc = balanced_accuracy_score(y_true_slides, y_pred_slides)
        
        log_output(f"Slide-level accuracy: {slide_accuracy:.4f}")
        log_output(f"Slide-level balanced accuracy: {slide_bacc:.4f}")
        
        # --- 4.6. Choose Best Clinical Model ---
        best_clinical_model = max(clinical_results.keys(),
                                key=lambda k: clinical_results[k]['accuracy'])
        
        log_output(f"Best clinical model: {best_clinical_model.upper()} "
                 f"(Acc: {clinical_results[best_clinical_model]['accuracy']:.4f})")
        
        # --- 4.7. Late Fusion: Combine Pathology and Clinical Predictions ---
        log_output("\n--- Late Fusion: Combining Predictions ---")
        
        # Create a mapping from patient ID to prediction
        slide_prob_map = {sid: prob for sid, prob in zip(slide_ids, y_prob_slides)} if y_prob_slides else {}
        slide_pred_map = {sid: pred for sid, pred in zip(slide_ids, y_pred_slides)}
        slide_truth_map = {sid: truth for sid, truth in zip(slide_ids, y_true_slides)}
        
        # Find common patients in both modalities
        common_patients = []
        clinical_probs = []
        pathology_probs = []
        true_labels = []
        
        for i, patient_id in enumerate(patient_ids_clinical_test):
            if patient_id in slide_prob_map:
                common_patients.append(patient_id)
                
                # Get clinical probability
                if clinical_results[best_clinical_model]['probabilities'] is not None:
                    if clinical_results[best_clinical_model]['probabilities'].shape[1] == 2:
                        # Binary classification - use probability of positive class
                        clinical_prob = clinical_results[best_clinical_model]['probabilities'][i][1]
                    else:
                        clinical_prob = clinical_results[best_clinical_model]['predictions'][i]
                else:
                    clinical_prob = clinical_results[best_clinical_model]['predictions'][i]
                
                clinical_probs.append(clinical_prob)
                
                # Get pathology probability
                pathology_prob = slide_prob_map[patient_id]
                pathology_probs.append(pathology_prob)
                
                # Get true label
                true_label = y_clinical_test[i]  # Use clinical ground truth
                true_labels.append(true_label)
                
                # Verify that labels match between modalities (sanity check)
                if true_label != slide_truth_map[patient_id]:
                    log_output(f"Warning: Label mismatch for patient {patient_id} "
                             f"(Clinical: {true_label}, Pathology: {slide_truth_map[patient_id]})")
        
        log_output(f"Found {len(common_patients)} common patients for fusion")
        
        if len(common_patients) > 0:
            # Perform weighted fusion
            fused_probs = []
            for clin_p, path_p in zip(clinical_probs, pathology_probs):
                # Weight according to feature dimensions
                weighted_prob = clinical_weight * clin_p + img_weight * path_p
                fused_probs.append(weighted_prob)
            
            # Convert probabilities to predictions
            fused_preds = [1 if p >= 0.5 else 0 for p in fused_probs]
            
            # Evaluate fusion performance
            fusion_acc = accuracy_score(true_labels, fused_preds)
            fusion_bacc = balanced_accuracy_score(true_labels, fused_preds)
            
            # Calculate AUC if possible
            try:
                fusion_auc = roc_auc_score(true_labels, fused_probs)
                log_output(f"Fusion AUC: {fusion_auc:.4f}")
            except:
                fusion_auc = None
                log_output("Could not compute fusion AUC (possibly only one class present)")
            
            log_output(f"Fusion accuracy: {fusion_acc:.4f}")
            log_output(f"Fusion balanced accuracy: {fusion_bacc:.4f}")
            
            # --- 4.8. Compare Modalities on Common Patients ---
            # Clinical-only predictions on common patients
            clinical_only_preds = [1 if p >= 0.5 else 0 for p in clinical_probs]
            clinical_only_acc = accuracy_score(true_labels, clinical_only_preds)
            clinical_only_bacc = balanced_accuracy_score(true_labels, clinical_only_preds)
            
            # Pathology-only predictions on common patients
            pathology_only_preds = [1 if p >= 0.5 else 0 for p in pathology_probs]
            pathology_only_acc = accuracy_score(true_labels, pathology_only_preds)
            pathology_only_bacc = balanced_accuracy_score(true_labels, pathology_only_preds)
            
            log_output("\n=== Performance on Common Patients ===")
            log_output(f"Clinical-only accuracy: {clinical_only_acc:.4f}, balanced: {clinical_only_bacc:.4f}")
            log_output(f"Pathology-only accuracy: {pathology_only_acc:.4f}, balanced: {pathology_only_bacc:.4f}")
            log_output(f"Fusion accuracy: {fusion_acc:.4f}, balanced: {fusion_bacc:.4f}")
            
            # Store results for this fold
            fold_results.append({
                'fold': fold + 1,
                'n_common_patients': len(common_patients),
                'clinical_accuracy': clinical_only_acc,
                'clinical_bacc': clinical_only_bacc,
                'pathology_accuracy': pathology_only_acc,
                'pathology_bacc': pathology_only_bacc,
                'fusion_accuracy': fusion_acc,
                'fusion_bacc': fusion_bacc,
                'fusion_auc': fusion_auc
            })
        else:
            log_output("No common patients found for fusion in this fold!")
            
    # ------ 5. Summarize Results Across Folds ------
    if fold_results:
        log_output("\n=== Cross-Validation Summary ===")
        
        # Calculate mean and std of metrics
        metrics = ['clinical_accuracy', 'clinical_bacc', 
                   'pathology_accuracy', 'pathology_bacc', 
                   'fusion_accuracy', 'fusion_bacc', 'fusion_auc']
        
        summary = {metric: [] for metric in metrics}
        
        for result in fold_results:
            for metric in metrics:
                if result.get(metric) is not None:
                    summary[metric].append(result[metric])
        
        # Print summary statistics
        log_output("Performance across folds (mean ± std):")
        for metric in metrics:
            if summary[metric]:
                mean_val = np.mean(summary[metric])
                std_val = np.std(summary[metric])
                log_output(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv("k_fold_fusion_results.csv", index=False)
        log_output("Results saved to k_fold_fusion_results.csv")
    else:
        log_output("No valid results obtained across folds.")

if __name__ == "__main__":
    main()
