import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score,
    cohen_kappa_score, classification_report, confusion_matrix,
    f1_score
)
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import matplotlib.pyplot as plt
import seaborn as sns

# Silence repeated convergence warnings
simplefilter("ignore", category=ConvergenceWarning)

# Setup logging
log_file = "multimodal_fusion_early_results.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

# ─────────────────────── 1. Reproducibility ────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────── 2. Paths ─────────────────────────────────
RESP_PATH = Path("patches_mix_response_portal_tract")
NORESP_PATH = Path("patches_mix_no_response_portal_tract")

assert RESP_PATH.is_dir() and NORESP_PATH.is_dir(), "Check dataset paths!"

# ─────────────────────── 3. Collect slides ─────────────────────────
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

# ─────────────────────── 4. Dataset class ───────────────────────────
class PatchDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ─────────────────────── 5. Encoder & transform ─────────────────────
from uni import get_encoder
model, transform = get_encoder(enc_name="uni2-h", device=device)
log_output(f"Loaded UNI2-h encoder model on {device}")

# ─────────────────────── 6. DataLoaders ───────────────────────────
BATCH = 16
train_dataset = PatchDataset(train_data, transform)
test_dataset = PatchDataset(test_data, transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                             num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False,
                            num_workers=4)

# ─────────────────────── 7. Create image index CSV ─────────────────
def make_df(items):
    paths, labels = zip(*items)
    classes = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({"image_path": paths, "label": labels, "class": classes})

csv_out = Path("index_path_mix_portal_tract.csv")
pd.concat([make_df(train_data), make_df(test_data)]).to_csv(csv_out, index_label="idx")
log_output(f"Saved slide index → {csv_out}")

# ─────────────────────── 8. Extract Features Directly ──────────────
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

# ─────────────────────── 9. Load and Process Clinical Data ─────────────────
log_output("Loading and preprocessing clinical data...")

# Load clinical data
df = pd.read_csv('cleaned_dataset.csv')
df_cleaned = df.drop(columns=['Patient study id Biopsy #1'])

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
for col in cat_summary.columns:
    df_cleaned = impute_column(df_cleaned, col)

num_summary = df_cleaned.describe()
for col in num_summary.columns:
    df_cleaned = impute_cont_column(df_cleaned, col)

# Create patient_id column from index for merging
df_cleaned = df_cleaned.reset_index()
df_cleaned.rename(columns={'index': 'patient_id'}, inplace=True)
df_cleaned['patient_id'] = df_cleaned['patient_id'].astype(str)

# Prepare clinical features
X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"])
y_clinical = df_cleaned['RAI Classification Biopsy #2']
patient_ids = df_cleaned['patient_id']

# Normalize clinical features
scaler_clinical = StandardScaler()
X_clinical_normalized = scaler_clinical.fit_transform(X_clinical)

# Create a DataFrame with normalized features for easy merging
clinical_features_df = pd.DataFrame(X_clinical_normalized, columns=X_clinical.columns)
clinical_features_df['patient_id'] = patient_ids
clinical_features_df['label'] = y_clinical

log_output(f"Clinical features shape: {X_clinical_normalized.shape}")
log_output(f"Number of patients in clinical data: {len(patient_ids)}")

# ─────────────────────── 10. Extract Patient IDs from Image Paths ───────────
log_output("Extracting patient IDs from image paths...")

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

# ─────────────────────── 11. Model Definitions ─────────────────────────
def get_models():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
            random_state=SEED, n_jobs=-1
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
            probability=True, random_state=SEED
        ),
        'LogisticRegression': sk_LogisticRegression(
            C=0.1, penalty='l2', class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=SEED
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, max_features='sqrt', random_state=SEED
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='euclidean', n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', alpha=0.001, max_iter=1000, random_state=SEED
        )
    }

# Evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, prefix=""):
    # Scale features for models that benefit from scaling
    if isinstance(model, (SVC, sk_LogisticRegression, KNeighborsClassifier, MLPClassifier)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train model
    log_output(f"Training {prefix} model...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    if y_prob is not None:
        if y_prob.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multi-class classification
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc = None
    
    result = {
        f'{prefix}_accuracy': accuracy,
        f'{prefix}_balanced_accuracy': balanced_acc,
        f'{prefix}_f1_score': f1,
        f'{prefix}_auc': auc,
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    log_output(f"{prefix} - Accuracy: {accuracy:.4f}, Balanced Acc: {balanced_acc:.4f}, F1: {f1:.4f}")
    if auc is not None:
        log_output(f"{prefix} - AUC: {auc:.4f}")
    
    return result

# ─────────────────────── 12. Option 1: Patch-level Fusion ─────────────────────
def patch_level_fusion():
    log_output("\n=== Option 1: Patch-level Fusion (Clinical Data Replication) ===")
    
    # 1. Create DataFrame with image embeddings
    train_img_df = pd.DataFrame(train_features['embeddings'])
    train_img_df['patient_id'] = train_imgs_df['patient_id'].values
    
    test_img_df = pd.DataFrame(test_features['embeddings'])
    test_img_df['patient_id'] = test_imgs_df['patient_id'].values
    
    # 2. Merge with clinical features (replicating clinical data for each patch)
    log_output("Merging image features with clinical data...")
    train_fused = pd.merge(
        train_img_df, 
        clinical_features_df.drop(columns=['label']), 
        on='patient_id', how='left'
    )
    
    test_fused = pd.merge(
        test_img_df, 
        clinical_features_df.drop(columns=['label']), 
        on='patient_id', how='left'
    )
    
    # 3. Handle missing values (patches without corresponding clinical data)
    missing_train = train_fused.isna().any(axis=1).sum()
    missing_test = test_fused.isna().any(axis=1).sum()
    
    if missing_train > 0 or missing_test > 0:
        log_output(f"Warning: {missing_train} train samples and {missing_test} test samples have missing clinical data")
        # Use only patches with matched clinical data
        train_fused = train_fused.dropna()
        test_fused = test_fused.dropna()
    
    # 4. Prepare features and labels
    X_train = train_fused.drop(columns=['patient_id']).values
    y_train = train_features['labels'][:len(X_train)]
    
    X_test = test_fused.drop(columns=['patient_id']).values
    y_test = test_features['labels'][:len(X_test)]
    
    log_output(f"Fused train features shape: {X_train.shape}")
    log_output(f"Fused test features shape: {X_test.shape}")
    
    # 5. Evaluate all models
    results = {}
    for model_name, model in get_models().items():
        log_output(f"Training {model_name}...")
        result = evaluate_model(model, X_train, y_train, X_test, y_test, prefix=f"patch_{model_name}")
        results[model_name] = result
    
    return results

# ─────────────────────── 13. Option 2: Patient-level Fusion ─────────────────────────
def patient_level_fusion():
    log_output("\n=== Option 2: Patient-level Fusion (Image Feature Aggregation) ===")
    
    # 1. Aggregate patch features per patient
    def aggregate_patient_features(image_features, imgs_df, method='mean'):
        """Aggregate multiple patches per patient into a single feature vector"""
        log_output(f"Aggregating patch features using {method} method...")
        # Create dataframe with features and patient IDs
        feat_cols = [f'feat_{i}' for i in range(image_features['embeddings'].shape[1])]
        df = pd.DataFrame(image_features['embeddings'], columns=feat_cols)
        df['patient_id'] = imgs_df['patient_id'].values
        df['label'] = image_features['labels']
        
        # Get a single label per patient (assuming all patches from a patient have same label)
        patient_labels = df.groupby('patient_id')['label'].first().reset_index()
        
        # Aggregate feature vectors for each patient
        if method == 'mean':
            patient_features = df.groupby('patient_id')[feat_cols].mean().reset_index()
        elif method == 'max':
            patient_features = df.groupby('patient_id')[feat_cols].max().reset_index()
        elif method == 'min':
            patient_features = df.groupby('patient_id')[feat_cols].min().reset_index()
        
        # Merge with labels
        patient_data = pd.merge(patient_features, patient_labels, on='patient_id')
        
        # Extract features and labels
        X = patient_data[feat_cols].values
        y = patient_data['label'].values
        patient_ids = patient_data['patient_id'].values
        
        return X, y, patient_ids, feat_cols
    
    # 2. Create patient-level features
    X_train_img, y_train_img, train_patient_ids, feat_cols = aggregate_patient_features(
        train_features, train_imgs_df, method='mean')
    X_test_img, y_test_img, test_patient_ids, _ = aggregate_patient_features(
        test_features, test_imgs_df, method='mean')
    
    log_output(f"Aggregated train image features shape: {X_train_img.shape}")
    log_output(f"Aggregated test image features shape: {X_test_img.shape}")
    log_output(f"Number of unique train patients after aggregation: {len(train_patient_ids)}")
    log_output(f"Number of unique test patients after aggregation: {len(test_patient_ids)}")
    
    # 3. Merge with clinical features
    # Create DataFrames for easier merging
    train_img_df = pd.DataFrame(X_train_img, columns=feat_cols)
    train_img_df['patient_id'] = train_patient_ids
    train_img_df['image_label'] = y_train_img
    
    test_img_df = pd.DataFrame(X_test_img, columns=feat_cols)
    test_img_df['patient_id'] = test_patient_ids
    test_img_df['image_label'] = y_test_img
    
    log_output("Merging aggregated patient features with clinical data...")
    # Merge with clinical features
    train_fused_df = pd.merge(
        train_img_df,
        clinical_features_df.drop(columns=['label']),
        on='patient_id',
        how='inner'
    )
    
    test_fused_df = pd.merge(
        test_img_df,
        clinical_features_df.drop(columns=['label']),
        on='patient_id',
        how='inner'
    )
    
    log_output(f"Number of patients after merging - Train: {len(train_fused_df)}, Test: {len(test_fused_df)}")
    
    # 4. Extract features and labels
    X_train_fused = train_fused_df.drop(columns=['patient_id', 'image_label']).values
    y_train_fused = train_fused_df['image_label'].values
    
    X_test_fused = test_fused_df.drop(columns=['patient_id', 'image_label']).values
    y_test_fused = test_fused_df['image_label'].values
    
    log_output(f"Fused train features shape: {X_train_fused.shape}")
    log_output(f"Fused test features shape: {X_test_fused.shape}")
    
    # 5. Evaluate all models
    results = {}
    for model_name, model in get_models().items():
        result = evaluate_model(model, X_train_fused, y_train_fused, X_test_fused, y_test_fused, 
                               prefix=f"patient_{model_name}")
        results[model_name] = result
    
    return results

# ─────────────────────── 14. Option 3: Enhanced Patient-level Fusion ─────────────────────────
def enhanced_patient_fusion():
    log_output("\n=== Option 3: Enhanced Patient-level Fusion (Multiple Aggregations) ===")
    
    def enhanced_patient_features(image_features, imgs_df):
        """Create rich patient-level features using multiple aggregation methods"""
        log_output("Creating enhanced patient-level features with multiple aggregations...")
        
        feat_cols = [f'feat_{i}' for i in range(image_features['embeddings'].shape[1])]
        df = pd.DataFrame(image_features['embeddings'], columns=feat_cols)
        df['patient_id'] = imgs_df['patient_id'].values
        df['label'] = image_features['labels']
        
        # Get labels per patient
        patient_labels = df.groupby('patient_id')['label'].first().reset_index()
        
        # 1. Calculate multiple aggregations
        patient_mean = df.groupby('patient_id')[feat_cols].mean()
        patient_max = df.groupby('patient_id')[feat_cols].max()
        patient_min = df.groupby('patient_id')[feat_cols].min()
        patient_std = df.groupby('patient_id')[feat_cols].std().fillna(0)  # Handle patients with single patch
        
        # 2. Combine all aggregations
        # Rename columns to avoid conflicts
        patient_mean.columns = [f'mean_{col}' for col in patient_mean.columns]
        patient_max.columns = [f'max_{col}' for col in patient_max.columns]
        patient_min.columns = [f'min_{col}' for col in patient_min.columns]
        patient_std.columns = [f'std_{col}' for col in patient_std.columns]
        
        # Join all features
        all_features = pd.concat([
            patient_mean.reset_index(),
            patient_max.reset_index().drop('patient_id', axis=1),
            patient_min.reset_index().drop('patient_id', axis=1),
            patient_std.reset_index().drop('patient_id', axis=1)
        ], axis=1)
        
        # 3. Add count of patches per patient as an additional feature
        patch_count = df.groupby('patient_id').size().reset_index(name='patch_count')
        all_features = pd.merge(all_features, patch_count, on='patient_id')
        
        # Merge with labels
        patient_data = pd.merge(all_features, patient_labels, on='patient_id')
        
        # Extract features, labels and IDs
        feature_cols = [col for col in patient_data.columns 
                      if col not in ['patient_id', 'label']]
        X = patient_data[feature_cols].values
        y = patient_data['label'].values
        patient_ids = patient_data['patient_id'].values
        
        return X, y, patient_ids, feature_cols
    
    # 1. Create enhanced patient features
    X_train_img, y_train_img, train_patient_ids, feat_cols = enhanced_patient_features(
        train_features, train_imgs_df)
    X_test_img, y_test_img, test_patient_ids, _ = enhanced_patient_features(
        test_features, test_imgs_df)
    
    log_output(f"Enhanced train image features shape: {X_train_img.shape}")
    log_output(f"Enhanced test image features shape: {X_test_img.shape}")
    log_output(f"Number of unique train patients: {len(train_patient_ids)}")
    log_output(f"Number of unique test patients: {len(test_patient_ids)}")
    
    # 2. Merge with clinical features
    # Create DataFrames for easier merging
    train_img_df = pd.DataFrame(X_train_img, columns=feat_cols)
    train_img_df['patient_id'] = train_patient_ids
    train_img_df['image_label'] = y_train_img
    
    test_img_df = pd.DataFrame(X_test_img, columns=feat_cols)
    test_img_df['patient_id'] = test_patient_ids
    test_img_df['image_label'] = y_test_img
    
    log_output("Merging enhanced patient features with clinical data...")
    # Merge with clinical features
    train_fused_df = pd.merge(
        train_img_df,
        clinical_features_df.drop(columns=['label']),
        on='patient_id',
        how='inner'
    )
    
    test_fused_df = pd.merge(
        test_img_df,
        clinical_features_df.drop(columns=['label']),
        on='patient_id',
        how='inner'
    )
    
    # 3. Extract features and labels
    X_train_fused = train_fused_df.drop(columns=['patient_id', 'image_label']).values
    y_train_fused = train_fused_df['image_label'].values
    
    X_test_fused = test_fused_df.drop(columns=['patient_id', 'image_label']).values
    y_test_fused = test_fused_df['image_label'].values
    
    log_output(f"Fused train features shape: {X_train_fused.shape}")
    log_output(f"Fused test features shape: {X_test_fused.shape}")
    
    # Optional: Feature selection to reduce dimensionality
    log_output("Applying feature selection...")
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=SEED), 
                             threshold='median')
    X_train_fused_selected = selector.fit_transform(X_train_fused, y_train_fused)
    X_test_fused_selected = selector.transform(X_test_fused)
    
    log_output(f"Original features: {X_train_fused.shape[1]}")
    log_output(f"Selected features: {X_train_fused_selected.shape[1]}")
    
    # 4. Evaluate all models
    results = {}
    for model_name, model in get_models().items():
        result = evaluate_model(model, X_train_fused_selected, y_train_fused, 
                             X_test_fused_selected, y_test_fused, 
                             prefix=f"enhanced_{model_name}")
        results[model_name] = result
    
    return results

# ─────────────────────── 15. Linear Probing (UNI) ─────────────────────────
def linear_probing():
    log_output("\n=== Linear Probing with UNI Features ===")
    
    # Import the UNI linear probing module
    from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
    
    # Evaluate linear probing
    metrics, dump = eval_linear_probe(
        train_feats=train_feats, train_labels=train_labels,
        valid_feats=None, valid_labels=None,
        test_feats=test_feats, test_labels=test_labels,
        max_iter=1000, verbose=False
    )
    
    log_output(f"Linear Probe Accuracy: {metrics.get('lin_acc', 0):.4f}")
    log_output(f"Linear Probe Balanced Accuracy: {metrics.get('lin_bacc', 0):.4f}")
    
    return {'LinearProbe': metrics}

# ─────────────────────── 16. Run All Experiments ─────────────────────────
log_output(f"\nStarting multimodal fusion experiments at {pd.Timestamp.now()}")

# Run experiments
option1_results = patch_level_fusion()
option2_results = patient_level_fusion()
option3_results = enhanced_patient_fusion()
linear_probe_results = linear_probing()

# ─────────────────────── 17. Compare and Visualize Results ─────────────────────────
log_output("\n=== Results Comparison ===")

# Create a summary DataFrame
summary_rows = []

for model_name, results in option1_results.items():
    summary_rows.append({
        'Method': f"Patch-level: {model_name}",
        'Accuracy': results[f'patch_{model_name}_accuracy'],
        'Balanced_Accuracy': results[f'patch_{model_name}_balanced_accuracy'],
        'AUC': results[f'patch_{model_name}_auc'] if results[f'patch_{model_name}_auc'] is not None else np.nan,
        'Fusion_Type': 'Patch-level'
    })

for model_name, results in option2_results.items():
    summary_rows.append({
        'Method': f"Patient-level: {model_name}",
        'Accuracy': results[f'patient_{model_name}_accuracy'],
        'Balanced_Accuracy': results[f'patient_{model_name}_balanced_accuracy'],
        'AUC': results[f'patient_{model_name}_auc'] if results[f'patient_{model_name}_auc'] is not None else np.nan,
        'Fusion_Type': 'Patient-level'
    })

for model_name, results in option3_results.items():
    summary_rows.append({
        'Method': f"Enhanced: {model_name}",
        'Accuracy': results[f'enhanced_{model_name}_accuracy'],
        'Balanced_Accuracy': results[f'enhanced_{model_name}_balanced_accuracy'],
        'AUC': results[f'enhanced_{model_name}_auc'] if results[f'enhanced_{model_name}_auc'] is not None else np.nan,
        'Fusion_Type': 'Enhanced'
    })

# Add linear probe results
summary_rows.append({
    'Method': "Linear Probe (UNI)",
    'Accuracy': linear_probe_results['LinearProbe'].get('lin_acc', 0),
    'Balanced_Accuracy': linear_probe_results['LinearProbe'].get('lin_bacc', 0),
    'AUC': np.nan,
    'Fusion_Type': 'Image-only'
})

# Create summary DataFrame and sort by accuracy
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('Accuracy', ascending=False)

# Save results to CSV
summary_df.to_csv('multimodal_fusion_results.csv', index=False)
log_output("Results saved to multimodal_fusion_results.csv")

# Print top 5 methods
log_output("\nTop 5 Methods:")
for i, row in summary_df.head(5).iterrows():
    log_output(f"{row['Method']} - Accuracy: {row['Accuracy']:.4f}, Balanced Acc: {row['Balanced_Accuracy']:.4f}")

# Visualize results
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")

# Plot bar chart with color-coded fusion types
ax = sns.barplot(
    x='Accuracy', 
    y='Method',
    hue='Fusion_Type',
    palette={
        'Patch-level': 'skyblue', 
        'Patient-level': 'lightgreen', 
        'Enhanced': 'coral',
        'Image-only': 'lightgray'
    },
    data=summary_df
)

plt.title('Multimodal Fusion Methods Performance Comparison', fontsize=16)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Method', fontsize=14)
plt.tight_layout()
plt.savefig('multimodal_fusion_comparison.png', dpi=300, bbox_inches='tight')

# Find the best model overall
best_method = summary_df.iloc[0]['Method']
best_accuracy = summary_df.iloc[0]['Accuracy']
best_fusion_type = summary_df.iloc[0]['Fusion_Type']

log_output(f"\nBest overall method: {best_method}")
log_output(f"Best accuracy: {best_accuracy:.4f}")
log_output(f"Best fusion type: {best_fusion_type}")
log_output(f"Analysis complete at {pd.Timestamp.now()}")