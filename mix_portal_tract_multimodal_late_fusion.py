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

# Silence repeated convergence warnings
simplefilter("ignore", category=ConvergenceWarning)

# Setup logging
log_file = "multimodal_late_fusion_results.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

# Clear log file if it exists
with open(log_file, "w") as f:
    f.write(f"===== Multimodal Late Fusion Results - {pd.Timestamp.now()} =====\n\n")

# ──────────────────────────── 1. Reproducibility ─────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_output(f"Using device: {device}")

# ──────────────────────────── 2. Paths ───────────────────────────────────────
RESP_PATH = Path("patches_mix_response_portal_tract")
NORESP_PATH = Path("patches_mix_no_response_portal_tract")

assert RESP_PATH.is_dir() and NORESP_PATH.is_dir(), "Check dataset paths!"

# ──────────────────────────── 3. Collect slides ──────────────────────────────
def collect(folder: Path, label: int):
    return [(str(folder / f), label) for f in os.listdir(folder)
            if f.lower().endswith(("_mix.png", ".jpg", ".jpeg"))]

all_images = collect(RESP_PATH, 0) + collect(NORESP_PATH, 1)

train_data, test_data = train_test_split(
    all_images, test_size=0.2, random_state=SEED,
    stratify=[lbl for _, lbl in all_images]
)

log_output(f"Total images: {len(all_images)}")
log_output(f"Training images: {len(train_data)}")
log_output(f"Testing images: {len(test_data)}")

# ──────────────────────────── 4. Dataset class ───────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items          # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ──────────────────────────── 5. Encoder & transform ─────────────────────────
log_output("Loading UNI encoder...")
from uni import get_encoder
model, transform = get_encoder(enc_name="uni2-h", device=device)
log_output("UNI encoder loaded successfully.")

# ──────────────────────────── 6. DataLoaders ────────────────────────────────
BATCH = 16
train_dataset = PatchDataset(train_data, transform)
test_dataset = PatchDataset(test_data, transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

# ──────────────────────────── 7. Create image index CSV ──────────────────────
def make_df(items):
    paths, labels = zip(*items)
    classes = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({"image_path": paths, "label": labels, "class": classes})

csv_out = Path("index_path_mix_portal_tract.csv")
pd.concat([make_df(train_data), make_df(test_data)]).to_csv(csv_out, index_label="idx")
log_output(f"Saved image index to {csv_out}")

# ──────────────────────────── 8. Extract Image Features ─────────────────────
log_output("Extracting features from training images...")
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
train_features = extract_patch_features_from_dataloader(model, train_dataloader)

log_output("Extracting features from testing images...")
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

# Convert to tensors and numpy arrays for different models
train_feats = torch.Tensor(train_features['embeddings'])
train_labels = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels = torch.Tensor(test_features['labels']).type(torch.long)

log_output(f"Train features shape: {train_feats.shape}")
log_output(f"Test features shape: {test_feats.shape}")

# Extract patient IDs from image paths
def extract_patient_id(img_path):
    basename = os.path.basename(img_path)
    parts = basename.split('-')
    if len(parts) >= 2:
        return parts[1]  # Assuming patient ID is in the second part
    return None

# Create DataFrames with image paths and patient IDs
train_imgs_df = pd.DataFrame(train_data, columns=['path', 'label'])
train_imgs_df['patient_id'] = train_imgs_df['path'].apply(extract_patient_id)

test_imgs_df = pd.DataFrame(test_data, columns=['path', 'label'])
test_imgs_df['patient_id'] = test_imgs_df['path'].apply(extract_patient_id)

# Count unique patients
n_train_patients = train_imgs_df['patient_id'].nunique()
n_test_patients = test_imgs_df['patient_id'].nunique()
log_output(f"Number of unique patients in training set: {n_train_patients}")
log_output(f"Number of unique patients in testing set: {n_test_patients}")

# ──────────────────────────── 9. Load Clinical Data ─────────────────────────
log_output("\n=== Loading and Processing Clinical Data ===")

df = pd.read_csv('cleaned_dataset.csv')
df_cleaned = df.rename(columns={'Patient study id Biopsy #1': 'patient_id'})

# Create age features
df_cleaned['Transplant Age'] = (((pd.to_datetime(df_cleaned['Date of transplant']) - 
                                pd.to_datetime(df_cleaned['Date of birth '])).dt.days)/365).astype(float)
df_cleaned['Biopsy Age'] = (((pd.to_datetime(df_cleaned['Biopsy (accession) Date Biopsy #1']) - 
                            pd.to_datetime(df_cleaned['Date of birth '])).dt.days)/365).astype(float)
df_cleaned['Transplant Biopsy Diff'] = (((pd.to_datetime(df_cleaned['Biopsy (accession) Date Biopsy #1']) - 
                                        pd.to_datetime(df_cleaned['Date of transplant'])).dt.days)/365).astype(float)

# Drop date columns
df_cleaned.drop(columns=['Date of birth ', 'Date of transplant', 'Biopsy (accession) Date Biopsy #1'], inplace=True)

cat_summary = df_cleaned.describe(include='object')

# Apply mappings for categorical variables
mappings = {
    'RAI Classification Biopsy #2': {'Response': 1, 'No Response': 0},
    'Gender': {'F': 0, 'M': 1},
}

for column, mapping in mappings.items():
    df_cleaned[column] = df_cleaned[column].map(mapping)

# Imputation functions
def impute_column(df, col):
    df_missing = df[df[col].isnull()]
    df_not_missing = df[df[col].notnull()]
    
    if df_missing.empty:
        return df
    
    X = df_not_missing.drop(columns=[col])
    y = df_not_missing[col]
    X_missing = df_missing.drop(columns=[col])
    
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X, y)
    df.loc[df[col].isnull(), col] = model.predict(X_missing)
    
    return df

def impute_cont_column(df, col):
    df_missing = df[df[col].isnull()]
    df_not_missing = df[df[col].notnull()]
    
    if df_missing.empty:
        return df
    
    X = df_not_missing.drop(columns=[col])
    y = df_not_missing[col]
    X_missing = df_missing.drop(columns=[col])
    
    model = RandomForestRegressor(random_state=SEED)
    model.fit(X, y)
    df.loc[df[col].isnull(), col] = model.predict(X_missing)
    
    return df

# Apply imputation
cat_summary = df_cleaned.describe(include='object')
for col in cat_summary.columns:
    df_cleaned = impute_column(df_cleaned, col)

num_summary = df_cleaned.describe()
for col in num_summary.columns:
    df_cleaned = impute_cont_column(df_cleaned, col)

# Create patient_id column from index for merging
df_cleaned['patient_id'] = df_cleaned['patient_id'].astype(str)

# Prepare clinical features
X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"])
y_clinical = df_cleaned['RAI Classification Biopsy #2']
patient_ids = df_cleaned['patient_id']

# Normalize clinical features
scaler_clinical = StandardScaler()
X_clinical_normalized = scaler_clinical.fit_transform(X_clinical)

log_output(f"Clinical data shape: {X_clinical_normalized.shape}")
log_output(f"Number of patients in clinical data: {len(patient_ids)}")

# ──────────────────────────── 10. Patient Matching ──────────────────────────
# Split clinical data independently into training and testing sets (80/20)
log_output("\n=== Splitting Clinical Data ===")

X_clinical_train, X_clinical_test, y_clinical_train, y_clinical_test, clinical_train_ids, clinical_test_ids = train_test_split(
    X_clinical_normalized, y_clinical, patient_ids, test_size=0.2, random_state=SEED, stratify=y_clinical
)

log_output(f"Clinical train data shape: {X_clinical_train.shape} ({len(clinical_train_ids)} patients)")
log_output(f"Clinical test data shape: {X_clinical_test.shape} ({len(clinical_test_ids)} patients)")

# Aggregated patient-level features for image data
def aggregate_patient_features(image_features, imgs_df):
    # Create dataframe with features and patient IDs
    feat_cols = list(range(image_features['embeddings'].shape[1]))
    df = pd.DataFrame(image_features['embeddings'], columns=feat_cols)
    df['patient_id'] = imgs_df['patient_id'].values
    df['label'] = image_features['labels']
    
    # Get a single label per patient
    patient_labels = df.groupby('patient_id')['label'].first().reset_index()
    
    # Mean pooling for patient-level features
    patient_features = df.groupby('patient_id')[feat_cols].mean().reset_index()
    
    # Merge with labels
    patient_data = pd.merge(patient_features, patient_labels, on='patient_id')
    
    # Extract features and labels
    X = patient_data[feat_cols].values
    y = patient_data['label'].values
    patient_ids = patient_data['patient_id'].values
    
    return X, y, patient_ids

# Get patient-level features for image data (for evaluation)
X_img_train, y_img_train, img_train_patients = aggregate_patient_features(train_features, train_imgs_df)
X_img_test, y_img_test, img_test_patients = aggregate_patient_features(test_features, test_imgs_df)

log_output(f"Aggregated image train data: {len(X_img_train)} patients")
log_output(f"Aggregated image test data: {len(X_img_test)} patients")

# ──────────────────────────── 11. Image Models Training ───────────────────────
log_output("\n=== Training Image Models ===")

# Function to evaluate a sklearn classifier
def train_eval_sklearn_model(model, X_train, y_train, X_test, y_test, model_name, scale_features=False):
    log_output(f"Training {model_name}...")
    
    # Scale features if needed
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    if y_prob is not None and y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
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

# 1. Random Forest on Image Features
rf_img = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10,
    min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
    random_state=SEED, n_jobs=-1
)
rf_img_results = train_eval_sklearn_model(
    rf_img, train_feats.numpy(), train_labels.numpy(), 
    test_feats.numpy(), test_labels.numpy(), "RandomForest (Image)", scale_features=False
)

# 2. SVM on Image Features
svm_img = SVC(
    C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
    probability=True, random_state=SEED
)
svm_img_results = train_eval_sklearn_model(
    svm_img, train_feats.numpy(), train_labels.numpy(), 
    test_feats.numpy(), test_labels.numpy(), "SVM (Image)", scale_features=True
)

# 3. Gradient Boosting on Image Features
gb_img = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=6,
    subsample=0.8, max_features='sqrt', random_state=SEED
)
gb_img_results = train_eval_sklearn_model(
    gb_img, train_feats.numpy(), train_labels.numpy(), 
    test_feats.numpy(), test_labels.numpy(), "GradientBoosting (Image)", scale_features=False
)

# 4. Linear Probing on Image Features
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
log_output("Training Linear Probe (UNI)...")
linprobe_metrics, linprobe_dump = eval_linear_probe(
    train_feats=train_feats, train_labels=train_labels,
    valid_feats=None, valid_labels=None,
    test_feats=test_feats, test_labels=test_labels,
    max_iter=1000, verbose=False
)
log_output(f"Linear Probe - Accuracy: {linprobe_metrics['lin_acc']:.4f}, Balanced Acc: {linprobe_metrics['lin_bacc']:.4f}")

# 5. KNN Probing on Image Features
from uni.downstream.eval_patch_features.fewshot import eval_knn
log_output("Training KNN Probe (UNI)...")
knn_metrics, knn_dump, _, _ = eval_knn(
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    center_feats=True, normalize_feats=True, n_neighbors=7
)
log_output(f"KNN Probe - Accuracy: {knn_metrics['knn7_acc']:.4f}, Balanced Acc: {knn_metrics['knn7_bacc']:.4f}")

# ──────────────────────────── 12. Clinical Models Training ─────────────────────
log_output("\n=== Training Clinical Models ===")

# 1. Random Forest on Clinical Features
rf_clinical = RandomForestClassifier(**{'random_state': 777, 'criterion': 'gini', 'n_estimators': 50, 'max_depth': 4})
rf_clinical_results = train_eval_sklearn_model(
    rf_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "RandomForest (Clinical)", scale_features=False
)

# 2. SVM on Clinical Features
svm_clinical = SVC(**{'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto', 'random_state': 8888}, probability=True)
svm_clinical_results = train_eval_sklearn_model(
    svm_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "SVM (Clinical)", scale_features=True
)

# 3. Gradient Boosting on Clinical Features
gb_clinical = GradientBoostingClassifier(**{'learning_rate': 0.1, 'n_estimators': 150, 'max_depth': 4, 'random_state': 2022})
gb_clinical_results = train_eval_sklearn_model(
    gb_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "GradientBoosting (Clinical)", scale_features=False
)

# 4. Logistic Regression on Clinical Features
lr_clinical = sk_LogisticRegression(**{'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'random_state': 9999})
lr_clinical_results = train_eval_sklearn_model(
    lr_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "LogisticRegression (Clinical)", scale_features=True
)

# ──────────────────────────── 13. Late Fusion Strategies ─────────────────────
log_output("\n=== Implementing Late Fusion Strategies ===")

# Function to create matched predictions for each patient
def get_patient_predictions(patient_ids, img_results, clinical_results):
    """
    Aligns predictions from image models and clinical models by patient ID,
    returning only patients present in both datasets.
    """
    # Extract patient-level predictions from image models
    patient_preds_img = {}
    for i, pid in enumerate(img_results['patient_ids']):
        patient_preds_img[pid] = {
            'true_label': img_results['true_labels'][i],
            'predictions': {model: img_results[model]['predictions'][i] for model in img_results if model != 'patient_ids' and model != 'true_labels'},
            'probabilities': {model: img_results[model]['probabilities'][i] if img_results[model]['probabilities'] is not None else None 
                               for model in img_results if model != 'patient_ids' and model != 'true_labels'}
        }
    
    # Extract predictions from clinical models
    patient_preds_clinical = {}
    for i, pid in enumerate(patient_ids):
        if pid in patient_preds_img:  # Only include patients present in image data
            patient_preds_clinical[pid] = {
                'predictions': {model: clinical_results[model]['predictions'][i] for model in clinical_results},
                'probabilities': {model: clinical_results[model]['probabilities'][i] if clinical_results[model]['probabilities'] is not None else None 
                                  for model in clinical_results}
            }
    
    # Combine predictions for patients present in both datasets
    matched_patients = []
    for pid in patient_preds_clinical:
        if pid in patient_preds_img:
            matched_patients.append({
                'patient_id': pid,
                'true_label': patient_preds_img[pid]['true_label'],
                'img_predictions': patient_preds_img[pid]['predictions'],
                'img_probabilities': patient_preds_img[pid]['probabilities'],
                'clinical_predictions': patient_preds_clinical[pid]['predictions'],
                'clinical_probabilities': patient_preds_clinical[pid]['probabilities']
            })
    
    return matched_patients

# Get patient-level predictions for test data
# First, aggregate patch-level image predictions by patient
def aggregate_image_predictions(patient_ids, patches_to_patients, predictions, probabilities=None):
    """Aggregate patch-level predictions to patient-level using majority voting"""
    patient_preds = defaultdict(list)
    patient_probs = defaultdict(list)
    
    for i, pred in enumerate(predictions):
        if i < len(patches_to_patients):
            patient = patches_to_patients[i]
            patient_preds[patient].append(pred)
            if probabilities is not None:
                patient_probs[patient].append(probabilities[i])
    
    # Majority vote for predictions
    agg_preds = []
    agg_probs = []
    for patient in patient_ids:
        if patient in patient_preds:
            # Majority vote for predictions
            agg_preds.append(np.round(np.mean(patient_preds[patient])).astype(int))
            
            # Average probabilities if available
            if probabilities is not None and patient in patient_probs:
                agg_probs.append(np.mean(patient_probs[patient], axis=0))
            else:
                agg_probs.append(None)
        else:
            # If patient has no predictions, use default (0)
            agg_preds.append(0)
            agg_probs.append(None)
    
    return agg_preds, agg_probs

# Create a mapping from patch index to patient
test_patches_to_patients = test_imgs_df['patient_id'].values
test_patient_ids = img_test_patients

# Create dictionary to store patient-level predictions for each image model
image_models_patient_results = {
    'patient_ids': test_patient_ids,
    'true_labels': y_img_test,
    'RandomForest': {
        'predictions': rf_img_results['predictions'],
        'probabilities': rf_img_results['probabilities']
    },
    'SVM': {
        'predictions': svm_img_results['predictions'],
        'probabilities': svm_img_results['probabilities']
    },
    'GradientBoosting': {
        'predictions': gb_img_results['predictions'],
        'probabilities': gb_img_results['probabilities']
    },
    'LinearProbe': {
        'predictions': linprobe_dump['preds_all'],
        'probabilities': None
    },
    'KNNProbe': {
        'predictions': knn_dump['preds_all'],
        'probabilities': knn_dump['probs_all'] if 'probs_all' in knn_dump else None
    }
}

# Create dictionary to store clinical model results
clinical_models_results = {
    'RandomForest': rf_clinical_results,
    'SVM': svm_clinical_results,
    'GradientBoosting': gb_clinical_results,
    'LogisticRegression': lr_clinical_results
}

# Match patients between image and clinical data
matched_patients = get_patient_predictions(test_patient_ids, image_models_patient_results, clinical_models_results)
log_output(f"Number of matched patients: {len(matched_patients)}")

# ──────────────────────── 14. Implement Late Fusion Strategies ───────────────
# Implementation of different late fusion strategies

# 1. Majority Voting across all models
def majority_voting_fusion(matched_patients):
    """Simple majority voting across all models"""
    y_true = []
    y_pred = []
    
    for patient in matched_patients:
        # Collect all predictions for this patient
        all_preds = []
        # Add image model predictions
        for model in patient['img_predictions']:
            all_preds.append(patient['img_predictions'][model])
        # Add clinical model predictions
        for model in patient['clinical_predictions']:
            all_preds.append(patient['clinical_predictions'][model])
        
        # Majority vote
        final_pred = np.round(np.mean(all_preds)).astype(int)
        
        y_true.append(patient['true_label'])
        y_pred.append(final_pred)
    
    return np.array(y_true), np.array(y_pred)

# 2. Weighted Voting (Image models have more weight)
def weighted_voting_fusion(matched_patients, image_weight=0.7):
    """Weighted voting between image models and clinical models"""
    y_true = []
    y_pred = []
    
    for patient in matched_patients:
        # Average predictions from image models
        img_preds = [patient['img_predictions'][model] for model in patient['img_predictions']]
        img_avg = np.mean(img_preds)
        
        # Average predictions from clinical models
        clinical_preds = [patient['clinical_predictions'][model] for model in patient['clinical_predictions']]
        clinical_avg = np.mean(clinical_preds)
        
        # Weighted average
        weighted_avg = image_weight * img_avg + (1 - image_weight) * clinical_avg
        final_pred = 1 if weighted_avg >= 0.5 else 0
        
        y_true.append(patient['true_label'])
        y_pred.append(final_pred)
    
    return np.array(y_true), np.array(y_pred)

# 3. Average Probabilities
def average_prob_fusion(matched_patients):
    """Average probabilities across models that provide probability outputs"""
    y_true = []
    y_pred = []
    y_prob = []
    
    for patient in matched_patients:
        # Collect all valid probabilities for positive class
        all_probs = []
        
        # Add image model probabilities
        for model in patient['img_probabilities']:
            prob = patient['img_probabilities'][model]
            if prob is not None and len(prob) == 2:  # Binary classification probabilities
                all_probs.append(prob[1])  # Probability of positive class
        
        # Add clinical model probabilities
        for model in patient['clinical_probabilities']:
            prob = patient['clinical_probabilities'][model]
            if prob is not None and len(prob) == 2:  # Binary classification probabilities
                all_probs.append(prob[1])  # Probability of positive class
        
        # Average probabilities if any valid probabilities exist
        if all_probs:
            avg_prob = np.mean(all_probs)
            final_pred = 1 if avg_prob >= 0.5 else 0
            
            y_true.append(patient['true_label'])
            y_pred.append(final_pred)
            y_prob.append(avg_prob)
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# 4. Best from each modality then majority vote
def best_models_fusion(matched_patients, best_img_model='RandomForest', best_clinical_model='RandomForest'):
    """Use only the best model from each modality"""
    y_true = []
    y_pred = []
    
    for patient in matched_patients:
        # Get prediction from best image model
        img_pred = patient['img_predictions'][best_img_model]
        
        # Get prediction from best clinical model
        clinical_pred = patient['clinical_predictions'][best_clinical_model]
        
        # Majority vote between the two best models
        final_pred = 1 if (img_pred + clinical_pred) >= 1 else 0
        
        y_true.append(patient['true_label'])
        y_pred.append(final_pred)
    
    return np.array(y_true), np.array(y_pred)

# 5. Meta-learner (stacking) with cross-validation
def meta_learner_fusion(matched_patients):
    """Train a meta-classifier using predictions from all models as features"""
    # Prepare data for meta-learner
    X_meta = []
    y_meta = []
    
    # Convert predictions to features for meta-learner
    for patient in matched_patients:
        features = []
        
        # Add image model predictions as features
        for model in sorted(patient['img_predictions'].keys()):
            features.append(patient['img_predictions'][model])
        
        # Add clinical model predictions as features
        for model in sorted(patient['clinical_predictions'].keys()):
            features.append(patient['clinical_predictions'][model])
        
        X_meta.append(features)
        y_meta.append(patient['true_label'])
    
    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)
    
    # Use leave-one-out cross-validation for meta-learner
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    
    # Meta-learner model
    meta_model = sk_LogisticRegression(C=1.0, class_weight='balanced', random_state=SEED)
    
    # Predictions with leave-one-out validation
    meta_preds = []
    for train_idx, test_idx in loo.split(X_meta):
        X_train, X_test = X_meta[train_idx], X_meta[test_idx]
        y_train = y_meta[train_idx]
        
        meta_model.fit(X_train, y_train)
        pred = meta_model.predict(X_test)[0]
        meta_preds.append(pred)
    
    return y_meta, np.array(meta_preds)

# ────────────────────── 15. Execute Fusion Strategies ────────────────────────
log_output("\n=== Executing Late Fusion Strategies ===")

# 1. Majority Voting
log_output("Applying Majority Voting fusion...")
y_true_mv, y_pred_mv = majority_voting_fusion(matched_patients)
acc_mv = accuracy_score(y_true_mv, y_pred_mv)
bacc_mv = balanced_accuracy_score(y_true_mv, y_pred_mv)
log_output(f"Majority Voting - Accuracy: {acc_mv:.4f}, Balanced Accuracy: {bacc_mv:.4f}")

# 2. Weighted Voting
log_output("Applying Weighted Voting fusion...")
y_true_wv, y_pred_wv = weighted_voting_fusion(matched_patients, image_weight=0.7)
acc_wv = accuracy_score(y_true_wv, y_pred_wv)
bacc_wv = balanced_accuracy_score(y_true_wv, y_pred_wv)
log_output(f"Weighted Voting (Image: 0.7, Clinical: 0.3) - Accuracy: {acc_wv:.4f}, Balanced Accuracy: {bacc_wv:.4f}")

# 3. Average Probabilities
log_output("Applying Average Probabilities fusion...")
y_true_ap, y_pred_ap, y_prob_ap = average_prob_fusion(matched_patients)
acc_ap = accuracy_score(y_true_ap, y_pred_ap)
bacc_ap = balanced_accuracy_score(y_true_ap, y_pred_ap)
auc_ap = roc_auc_score(y_true_ap, y_prob_ap)
log_output(f"Average Probabilities - Accuracy: {acc_ap:.4f}, Balanced Accuracy: {bacc_ap:.4f}, AUC: {auc_ap:.4f}")

# 4. Best Models Fusion
best_img_model = max(
    ['RandomForest', 'SVM', 'GradientBoosting', 'LinearProbe', 'KNNProbe'],
    key=lambda m: accuracy_score(y_img_test, image_models_patient_results[m]['predictions'])
)
best_clinical_model = max(
    ['RandomForest', 'SVM', 'GradientBoosting', 'LogisticRegression'],
    key=lambda m: clinical_models_results[m]['accuracy']
)

log_output(f"Best image model: {best_img_model}")
log_output(f"Best clinical model: {best_clinical_model}")

log_output("Applying Best Models fusion...")
y_true_bm, y_pred_bm = best_models_fusion(matched_patients, best_img_model, best_clinical_model)
acc_bm = accuracy_score(y_true_bm, y_pred_bm)
bacc_bm = balanced_accuracy_score(y_true_bm, y_pred_bm)
log_output(f"Best Models - Accuracy: {acc_bm:.4f}, Balanced Accuracy: {bacc_bm:.4f}")

# 5. Meta-learner
log_output("Applying Meta-learner fusion...")
y_true_ml, y_pred_ml = meta_learner_fusion(matched_patients)
acc_ml = accuracy_score(y_true_ml, y_pred_ml)
bacc_ml = balanced_accuracy_score(y_true_ml, y_pred_ml)
log_output(f"Meta-learner - Accuracy: {acc_ml:.4f}, Balanced Accuracy: {bacc_ml:.4f}")

# ────────────────────── 16. Summarize Results ────────────────────────────────
log_output("\n=== Summary of Results ===")

# Individual modality performance (baseline)
log_output("\nIndividual Modality Performance:")

# Best image model
image_model_accs = {
    'RandomForest': accuracy_score(y_img_test, image_models_patient_results['RandomForest']['predictions']),
    'SVM': accuracy_score(y_img_test, image_models_patient_results['SVM']['predictions']),
    'GradientBoosting': accuracy_score(y_img_test, image_models_patient_results['GradientBoosting']['predictions']),
    'LinearProbe': accuracy_score(y_img_test, image_models_patient_results['LinearProbe']['predictions']),
    'KNNProbe': accuracy_score(y_img_test, image_models_patient_results['KNNProbe']['predictions'])
}
best_image_acc = max(image_model_accs.values())
log_output(f"Best Image Model: {max(image_model_accs, key=image_model_accs.get)} - Accuracy: {best_image_acc:.4f}")

# Best clinical model
clinical_model_accs = {
    'RandomForest': clinical_models_results['RandomForest']['accuracy'],
    'SVM': clinical_models_results['SVM']['accuracy'],
    'GradientBoosting': clinical_models_results['GradientBoosting']['accuracy'],
    'LogisticRegression': clinical_models_results['LogisticRegression']['accuracy']
}
best_clinical_acc = max(clinical_model_accs.values())
log_output(f"Best Clinical Model: {max(clinical_model_accs, key=clinical_model_accs.get)} - Accuracy: {best_clinical_acc:.4f}")

# Fusion strategies
log_output("\nFusion Strategies Performance:")
fusion_results = {
    'Majority Voting': {'accuracy': acc_mv, 'balanced_accuracy': bacc_mv},
    'Weighted Voting': {'accuracy': acc_wv, 'balanced_accuracy': bacc_wv},
    'Average Probabilities': {'accuracy': acc_ap, 'balanced_accuracy': bacc_ap, 'auc': auc_ap},
    'Best Models': {'accuracy': acc_bm, 'balanced_accuracy': bacc_bm},
    'Meta-learner': {'accuracy': acc_ml, 'balanced_accuracy': bacc_ml}
}

# Find best fusion strategy
best_fusion_acc = 0
best_fusion_method = ""
for method, metrics in fusion_results.items():
    if metrics['accuracy'] > best_fusion_acc:
        best_fusion_acc = metrics['accuracy']
        best_fusion_method = method
    log_output(f"{method} - Accuracy: {metrics['accuracy']:.4f}, Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if 'auc' in metrics:
        log_output(f"  AUC: {metrics['auc']:.4f}")

# Compare with single modality performance
log_output("\nComparison with Single Modality:")
if best_fusion_acc > max(best_image_acc, best_clinical_acc):
    log_output(f"Fusion improved performance! Best fusion ({best_fusion_method}): {best_fusion_acc:.4f} vs. Best single modality: {max(best_image_acc, best_clinical_acc):.4f}")
else:
    log_output(f"Fusion did not improve performance. Best fusion: {best_fusion_acc:.4f} vs. Best single modality: {max(best_image_acc, best_clinical_acc):.4f}")

# ────────────────────── 17. Create Visualizations ──────────────────────────────
log_output("\n=== Creating Visualizations ===")

# Confusion matrix for best fusion strategy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))

# Select the true and predicted labels from the best fusion method
if best_fusion_method == 'Majority Voting':
    y_true_best, y_pred_best = y_true_mv, y_pred_mv
elif best_fusion_method == 'Weighted Voting':
    y_true_best, y_pred_best = y_true_wv, y_pred_wv
elif best_fusion_method == 'Average Probabilities':
    y_true_best, y_pred_best = y_true_ap, y_pred_ap
elif best_fusion_method == 'Best Models':
    y_true_best, y_pred_best = y_true_bm, y_pred_bm
elif best_fusion_method == 'Meta-learner':
    y_true_best, y_pred_best = y_true_ml, y_pred_ml

cm = confusion_matrix(y_true_best, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Response', 'No Response'], 
            yticklabels=['Response', 'No Response'])
plt.title(f'Confusion Matrix for {best_fusion_method} Fusion')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f"confusion_matrix_{best_fusion_method.lower().replace(' ', '_')}.png", dpi=300)

# Performance comparison bar chart
plt.figure(figsize=(12, 8))

# Prepare data for the bar chart
methods = []
accuracies = []
balanced_accs = []

# Add individual model results
methods.extend([f"Image: {model}" for model in image_model_accs.keys()])
accuracies.extend(list(image_model_accs.values()))
balanced_accs.extend([balanced_accuracy_score(y_img_test, image_models_patient_results[model]['predictions']) 
                     for model in image_model_accs.keys()])

methods.extend([f"Clinical: {model}" for model in clinical_model_accs.keys()])
accuracies.extend(list(clinical_model_accs.values()))
balanced_accs.extend([clinical_models_results[model]['balanced_accuracy'] for model in clinical_model_accs.keys()])

# Add fusion results
methods.extend([f"Fusion: {method}" for method in fusion_results.keys()])
accuracies.extend([metrics['accuracy'] for metrics in fusion_results.values()])
balanced_accs.extend([metrics['balanced_accuracy'] for metrics in fusion_results.values()])

# Create DataFrame for plotting
results_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracies,
    'Balanced Accuracy': balanced_accs
})

# Determine the type of method (image, clinical, fusion)
method_type = []
for method in methods:
    if method.startswith("Image:"):
        method_type.append("Image")
    elif method.startswith("Clinical:"):
        method_type.append("Clinical")
    else:
        method_type.append("Fusion")
results_df['Type'] = method_type

# Sort by accuracy
results_df = results_df.sort_values('Accuracy', ascending=False)

# Plot
plt.figure(figsize=(14, 10))
ax = sns.barplot(x='Accuracy', y='Method', hue='Type', data=results_df, 
                palette={'Image': 'skyblue', 'Clinical': 'lightgreen', 'Fusion': 'coral'})
plt.title('Performance Comparison of All Models and Fusion Strategies', fontsize=16)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Method', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')

# Create a correlation matrix of all model predictions
plt.figure(figsize=(12, 10))

# Create a DataFrame with all model predictions
model_predictions = pd.DataFrame()

# Add image model predictions
for model in image_model_accs.keys():
    model_predictions[f"Img_{model}"] = image_models_patient_results[model]['predictions']

# Add clinical model predictions for matched patients
for i, patient in enumerate(matched_patients):
    for model in clinical_model_accs.keys():
        if i == 0:  # Initialize columns on first iteration
            model_predictions[f"Clin_{model}"] = None
        # Set the prediction for this patient
        model_predictions.at[i, f"Clin_{model}"] = patient['clinical_predictions'][model]

# Add fusion strategy predictions
model_predictions['Fusion_MV'] = y_pred_mv
model_predictions['Fusion_WV'] = y_pred_wv
model_predictions['Fusion_AP'] = y_pred_ap
model_predictions['Fusion_BM'] = y_pred_bm
model_predictions['Fusion_ML'] = y_pred_ml

# Add ground truth
model_predictions['True_Label'] = y_true_best

# Calculate correlation matrix
corr_matrix = model_predictions.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Model Predictions', fontsize=16)
plt.tight_layout()
plt.savefig("prediction_correlation_matrix.png", dpi=300, bbox_inches='tight')

# ────────────────────── 18. Save Final Results ─────────────────────────────────
log_output("\n=== Saving Final Results ===")

# Save results to CSV
results_df.to_csv("multimodal_late_fusion_results.csv", index=False)
log_output("Performance results saved to multimodal_late_fusion_results.csv")

# Save patient-level predictions
patient_predictions = pd.DataFrame({
    'patient_id': [p['patient_id'] for p in matched_patients],
    'true_label': y_true_best,
    'majority_voting': y_pred_mv,
    'weighted_voting': y_pred_wv,
    'average_probabilities': y_pred_ap,
    'best_models': y_pred_bm,
    'meta_learner': y_pred_ml
})
patient_predictions.to_csv("patient_level_fusion_predictions.csv", index=False)
log_output("Patient-level predictions saved to patient_level_fusion_predictions.csv")

log_output(f"\nMultimodal Late Fusion analysis complete at {pd.Timestamp.now()}")
log_output(f"Best fusion method: {best_fusion_method} with accuracy: {best_fusion_acc:.4f}")