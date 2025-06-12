import os
import random
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score,
    cohen_kappa_score, classification_report, confusion_matrix,
    f1_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from utils.my_utils import get_eval_metrics, eval_sklearn_classifier, calculate_feature_weights, get_slide_level_predictions, train_eval_sklearn_model

# Set up logging
log_file = "early_fusion_multimodal_hypertuning.log"
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


def extract_patient_id_from_path(path):
    """
    Extract patient ID from file path with improved error handling.
    
    Expected format: prefix-patientID-other_info.ext
    """
    try:
        basename = os.path.basename(path)
        # Remove file extension
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('-')
        
        if len(parts) >= 2:
            # More robust patient ID extraction
            patient_id = parts[1].strip()
            # Validate patient ID (should not be empty and should be alphanumeric)
            if patient_id and patient_id.replace('_', '').isalnum():
                return patient_id
        
        log_output(f"WARNING: Could not extract patient ID from path: {path}")
        return "unknown"
    except Exception as e:
        log_output(f"ERROR: Exception extracting patient ID from {path}: {e}")
        return "unknown"


def validate_patient_splits(train_patients, test_patients, description=""):
    """Validate that train and test patient sets don't overlap."""
    train_set = set(train_patients)
    test_set = set(test_patients)
    overlap = train_set.intersection(test_set)
    
    if overlap:
        log_output(f"❌ CRITICAL ERROR {description}: {len(overlap)} patients in both train and test!")
        log_output(f"   Overlapping patients: {list(overlap)[:10]}...")  # Show first 10
        return False
    else:
        log_output(f"✓ Patient split validation passed {description}")
        return True


def aggregate_pathology_features(paths, features, patient_ids, labels=None, min_patches=1):
    """
    Aggregate patch-level pathology features to patient-level features with validation.
    
    Args:
        paths: List of image paths
        features: Array of feature embeddings
        patient_ids: List of patient IDs extracted from paths
        labels: Optional array of labels for consistency checking
        min_patches: Minimum number of patches required per patient
        
    Returns:
        Dictionary mapping patient IDs to aggregated features and metadata
    """
    patient_data = defaultdict(lambda: {'features': [], 'labels': [], 'paths': []})
    
    # Group data by patient
    for i, (path, feature) in enumerate(zip(paths, features)):
        patient_id = patient_ids[i]
        if patient_id != "unknown":
            patient_data[patient_id]['features'].append(feature)
            patient_data[patient_id]['paths'].append(path)
            if labels is not None:
                patient_data[patient_id]['labels'].append(labels[i])
    
    # Aggregate and validate
    aggregated_results = {}
    patients_removed = []
    
    for patient_id, data in patient_data.items():
        feat_list = data['features']
        label_list = data['labels']
        
        # Check minimum patches requirement
        if len(feat_list) < min_patches:
            patients_removed.append(patient_id)
            continue
        
        # Check label consistency for this patient
        if label_list:
            unique_labels = set(label_list)
            if len(unique_labels) > 1:
                log_output(f"WARNING: Patient {patient_id} has inconsistent labels: {unique_labels}")
                # Use majority vote
                from collections import Counter
                label_counts = Counter(label_list)
                majority_label = label_counts.most_common(1)[0][0]
                log_output(f"   Using majority label: {majority_label}")
            else:
                majority_label = label_list[0]
        else:
            majority_label = None
            
        # Aggregate features (using mean)
        aggregated_features = np.mean(feat_list, axis=0)
        
        aggregated_results[patient_id] = {
            'features': aggregated_features,
            'label': majority_label,
            'num_patches': len(feat_list),
            'paths': data['paths']
        }
    
    if patients_removed:
        log_output(f"Removed {len(patients_removed)} patients with < {min_patches} patches")
    
    return aggregated_results


def create_multimodal_dataset(clinical_data, pathology_data, patient_ids_clinical, y_clinical):
    """
    Create multimodal dataset by matching clinical and pathology data.
    
    Returns matched data for patients that have both modalities.
    """
    matched_patients = []
    matched_clinical_features = []
    matched_pathology_features = []
    matched_labels = []
    matched_patch_counts = []
    
    # Create mapping from clinical patient ID to index and label
    clinical_patient_map = {}
    for i, patient_id in enumerate(patient_ids_clinical):
        if patient_id in clinical_patient_map:
            log_output(f"WARNING: Duplicate patient {patient_id} in clinical data")
        clinical_patient_map[patient_id] = {'index': i, 'label': y_clinical[i]}
    
    # Match with pathology data
    for patient_id, path_data in pathology_data.items():
        if patient_id in clinical_patient_map:
            clinical_info = clinical_patient_map[patient_id]
            
            # Verify label consistency
            clinical_label = clinical_info['label']
            pathology_label = path_data['label']
            
            if pathology_label is not None and clinical_label != pathology_label:
                log_output(f"WARNING: Label mismatch for patient {patient_id}: "
                          f"clinical={clinical_label}, pathology={pathology_label}")
            
            matched_patients.append(patient_id)
            matched_clinical_features.append(clinical_data[clinical_info['index']])
            matched_pathology_features.append(path_data['features'])
            matched_labels.append(clinical_label)  # Use clinical label as ground truth
            matched_patch_counts.append(path_data['num_patches'])
    
    return {
        'patients': matched_patients,
        'clinical_features': np.array(matched_clinical_features),
        'pathology_features': np.array(matched_pathology_features),
        'labels': np.array(matched_labels),
        'patch_counts': matched_patch_counts
    }


def get_hyperparameter_grids():
    """Define hyperparameter grids for all models."""
    
    # Clinical and Pathology model grids
    param_grids = {
        'lr': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'random_state': [42]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'random_state': [8888]
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, None],
            'criterion': ['gini', 'entropy'],
            'random_state': [777]
        },
        'gb': {
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'random_state': [32]
        }
    }
    
    # For early fusion, we use the same parameter grids as individual models
    # since we're just concatenating features and training regular classifiers
    fusion_param_grids = param_grids.copy()
    
    return param_grids, fusion_param_grids


def create_model_instances():
    """Create model instances for hyperparameter tuning."""
    models = {
        'lr': sk_LogisticRegression(max_iter=1000),
        'svm': SVC(probability=True),
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier()
    }
    return models


def create_fusion_model_instances():
    """Create fusion model instances for hyperparameter tuning."""
    # For early fusion, these are just regular models that will work on concatenated features
    models = {
        'lr': sk_LogisticRegression(max_iter=1000),
        'svm': SVC(probability=True),
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier()
    }
    return models


def evaluate_model_with_metrics(y_true, y_pred, y_proba=None):
    """Calculate all required metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = None
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'auc': auc
    }


def tune_hyperparameters_on_training_set(X_train, y_train, model_name, base_model, param_grid, inner_cv=3):
    """Tune hyperparameters using only the training data with inner cross-validation."""
    
    # Create pipeline with scaling for models that need it
    if model_name in ['lr', 'svm']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clone(base_model))
        ])
        # Adjust parameter names for pipeline
        param_grid_adjusted = {}
        for key, value in param_grid.items():
            param_grid_adjusted[f'classifier__{key}'] = value
    else:
        pipeline = clone(base_model)
        param_grid_adjusted = param_grid.copy()
    
    # Perform GridSearchCV on training data only
    grid_search = GridSearchCV(
        pipeline, 
        param_grid_adjusted, 
        cv=inner_cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        return best_model, best_params, best_score
    except Exception as e:
        log_output(f"❌ Error during hyperparameter tuning: {e}")
        return None, {}, 0.0


def visualize_aggregated_features(clinical_features, pathology_features, labels, patient_ids, output_dir="feature_visualizations"):
    """
    Create visualizations of aggregated features.
    
    Args:
        clinical_features: Array of clinical features per patient
        pathology_features: Array of aggregated pathology features per patient
        labels: Array of labels per patient
        patient_ids: List of patient IDs
        output_dir: Directory to save visualization outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    label_names = ["Response", "No Response"]
    
    log_output(f"\n=== Creating Feature Visualizations in {output_dir} ===")
    
    # 1. Feature distribution visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate mean and std for pathology features
    path_means = np.mean(pathology_features, axis=1)
    path_stds = np.std(pathology_features, axis=1)
    
    # Split by class
    class0_means = path_means[labels == 0]
    class1_means = path_means[labels == 1]
    class0_stds = path_stds[labels == 0]
    class1_stds = path_stds[labels == 1]
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(class0_means, label=label_names[0], shade=True)
    sns.kdeplot(class1_means, label=label_names[1], shade=True)
    plt.title('Distribution of Mean Feature Values')
    plt.xlabel('Mean Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(class0_stds, label=label_names[0], shade=True)
    sns.kdeplot(class1_stds, label=label_names[1], shade=True)
    plt.title('Distribution of Feature Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300)
    
    # 2. t-SNE visualization
    log_output("Applying t-SNE to visualize feature spaces...")
    
    # Function to create t-SNE plot
    def create_tsne_plot(features, title, filename):
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i, label in enumerate([0, 1]):
            indices = labels == label
            plt.scatter(
                features_2d[indices, 0], 
                features_2d[indices, 1],
                label=label_names[i],
                alpha=0.7
            )
        
        # Add patient IDs as annotations
        for i, pid in enumerate(patient_ids):
            plt.annotate(pid, (features_2d[i, 0], features_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300)
    
    # Create t-SNE plots for each feature type
    create_tsne_plot(clinical_features, 'Clinical Features t-SNE', f'{output_dir}/clinical_tsne.png')
    create_tsne_plot(pathology_features, 'Pathology Features t-SNE', f'{output_dir}/pathology_tsne.png')
    
    # Concatenate features for combined visualization
    combined_features = np.concatenate((clinical_features, pathology_features), axis=1)
    create_tsne_plot(combined_features, 'Combined Features t-SNE', f'{output_dir}/combined_tsne.png')
    
    # 3. PCA visualization for pathology features
    log_output("Creating PCA visualization for pathology features...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pathology_features)
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate([0, 1]):
        indices = labels == label
        plt.scatter(
            pca_result[indices, 0], 
            pca_result[indices, 1],
            label=label_names[i],
            alpha=0.7
        )
    
    # Add patient IDs
    for i, pid in enumerate(patient_ids):
        plt.annotate(pid, (pca_result[i, 0], pca_result[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'PCA of Pathology Features (explained variance: {pca.explained_variance_ratio_.sum():.2f})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pathology_pca.png', dpi=300)
    
    # 4. Feature heatmap for selected patients
    log_output("Creating feature heatmaps...")
    
    # Select a subset of patients for the heatmap (avoid overcrowding)
    max_patients = min(20, len(patient_ids))
    sample_indices = np.random.choice(len(patient_ids), max_patients, replace=False)
    
    # Clinical features heatmap
    plt.figure(figsize=(12, 10))
    clinical_subset = clinical_features[sample_indices]
    sns.heatmap(clinical_subset, cmap='viridis')
    plt.title('Clinical Features Heatmap (Sample of Patients)')
    plt.xlabel('Feature Index')
    plt.ylabel('Patient Index')
    plt.savefig(f'{output_dir}/clinical_heatmap.png', dpi=300)
    
    # Pathology features heatmap (sample due to high dimensionality)
    plt.figure(figsize=(14, 10))
    # Take first 100 pathology features for visualization
    pathology_subset = pathology_features[sample_indices, :100]
    sns.heatmap(pathology_subset, cmap='viridis')
    plt.title('Pathology Features Heatmap (First 100 features, Sample of Patients)')
    plt.xlabel('Feature Index')
    plt.ylabel('Patient Index')
    plt.savefig(f'{output_dir}/pathology_heatmap.png', dpi=300)
    
    log_output(f"Feature visualizations created and saved to directory: {output_dir}")

def visualize_patient_embeddings(clinical_features, pathology_features, labels, patient_ids, n_patients=10, output_dir="patient_visualizations"):
    """
    Visualize aggregated mean feature embeddings for random patients.
    
    Args:
        clinical_features: Array of clinical features per patient
        pathology_features: Array of pathology features per patient
        labels: Array of labels per patient
        patient_ids: List of patient IDs
        n_patients: Number of random patients to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    label_names = ["Response", "No Response"]
    
    # Select random patients, ensuring mixture of both classes if possible
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]
    
    n_class0 = min(n_patients // 2, len(class0_indices))
    n_class1 = min(n_patients - n_class0, len(class1_indices))
    
    # Adjust if one class has fewer patients
    if n_class0 + n_class1 < n_patients:
        if len(class0_indices) > n_class0:
            n_class0 = min(n_patients - n_class1, len(class0_indices))
        elif len(class1_indices) > n_class1:
            n_class1 = min(n_patients - n_class0, len(class1_indices))
    
    # Select random indices from each class
    selected_class0 = np.random.choice(class0_indices, n_class0, replace=False)
    selected_class1 = np.random.choice(class1_indices, n_class1, replace=False)
    selected_indices = np.concatenate([selected_class0, selected_class1])
    
    # Get selected data
    selected_patient_ids = [patient_ids[i] for i in selected_indices]
    selected_labels = labels[selected_indices]
    selected_clinical = clinical_features[selected_indices]
    selected_pathology = pathology_features[selected_indices]
    
    log_output(f"\n=== Visualizing Embeddings for {len(selected_indices)} Random Patients ===")
    
    # 1. Heatmap of pathology embeddings (first 100 dimensions)
    plt.figure(figsize=(16, 10))
    display_dims = min(100, selected_pathology.shape[1])  # Display up to 100 dimensions
    
    # Sort by class for better visualization
    sort_idx = np.argsort(selected_labels)
    sorted_ids = [selected_patient_ids[i] for i in sort_idx]
    sorted_labels = selected_labels[sort_idx]
    sorted_pathology = selected_pathology[sort_idx, :display_dims]
    
    # Create custom y-tick labels with patient ID and class
    y_labels = [f"{pid} ({label_names[label]})" for pid, label in zip(sorted_ids, sorted_labels)]
    
    # Plot the heatmap
    ax = sns.heatmap(sorted_pathology, cmap="viridis", yticklabels=y_labels)
    plt.title(f'Pathology Feature Embeddings for {len(selected_indices)} Patients (First {display_dims} dimensions)')
    plt.xlabel('Feature Dimension')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/patient_pathology_embeddings.png', dpi=300)
    
    # 2. Clinical feature comparison
    plt.figure(figsize=(14, 8))
    sorted_clinical = selected_clinical[sort_idx]
    
    sns.heatmap(sorted_clinical, cmap="coolwarm", yticklabels=y_labels)
    plt.title(f'Clinical Features for {len(selected_indices)} Patients')
    plt.xlabel('Clinical Feature Index')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/patient_clinical_features.png', dpi=300)
    
    # 3. Embedding statistics visualization
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics for each patient
    means = np.mean(selected_pathology, axis=1)
    stds = np.std(selected_pathology, axis=1)
    mins = np.min(selected_pathology, axis=1)
    maxs = np.max(selected_pathology, axis=1)
    
    # Create barplot of embedding statistics
    x = np.arange(len(selected_patient_ids))
    width = 0.2
    
    # Plot bars for each statistic
    plt.bar(x - width*1.5, means, width, label='Mean')
    plt.bar(x - width*0.5, stds, width, label='Std Dev')
    plt.bar(x + width*0.5, maxs, width, label='Max')
    plt.bar(x + width*1.5, mins, width, label='Min')
    
    # Add color bands to indicate class
    for i in range(len(selected_indices)):
        if selected_labels[i] == 0:
            plt.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='green')
        else:
            plt.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
    
    plt.xticks(x, selected_patient_ids, rotation=90)
    plt.legend()
    plt.title('Pathology Embedding Statistics by Patient')
    plt.ylabel('Value')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/patient_embedding_statistics.png', dpi=300)
    
    # 4. Feature distribution by patient
    plt.figure(figsize=(15, 10))
    
    # Create subplots, one for each patient
    rows = int(np.ceil(len(selected_indices) / 3))
    cols = min(3, len(selected_indices))
    
    for i, idx in enumerate(range(len(selected_indices))):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i+1)
        
        # Get patient data
        patient_id = selected_patient_ids[idx]
        patient_label = selected_labels[idx]
        patient_embedding = selected_pathology[idx]
        
        # Plot histogram of features
        sns.histplot(patient_embedding, kde=True)
        plt.title(f"Patient {patient_id}\n{label_names[patient_label]}")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/patient_embedding_distributions.png', dpi=300)
    
    log_output(f"✓ Patient-level visualizations saved to {output_dir}/")
    
    return selected_indices, selected_patient_ids

# Add visualization call in the main function after creating multimodal dataset
def main():
    log_output("=== Starting Nested Cross-Validation for Multimodal Early Fusion ===")
    log_output(f"Log file: {log_file}")
    log_output(f"Dataset info: 55 patients, 13 clinical features, 1536 pathology features")
    log_output(f"Patch info: 2272 Response + 1756 No Response = 4028 total patches (224x224px)")
    
    # ------ 1. Load Clinical Data ------
    log_output("\n=== Loading Clinical Data ===")
    with open('clinical_features_portal_tract.pkl', 'rb') as f:
        df_cleaned = pickle.load(f)
    
    X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"]).values
    y_clinical = df_cleaned['RAI Classification Biopsy #2'].values
    patient_ids_clinical = df_cleaned['patient_id'].values
    
    log_output(f"Clinical data shape: {X_clinical.shape}")
    log_output(f"Number of patients in clinical data: {len(patient_ids_clinical)}")
    log_output(f"Unique patients in clinical data: {len(set(patient_ids_clinical))}")
    
    # Check for duplicate patients in clinical data
    clinical_duplicates = [pid for pid, count in Counter(patient_ids_clinical).items() if count > 1]
    if clinical_duplicates:
        log_output(f"❌ CRITICAL: Found {len(clinical_duplicates)} duplicate patient IDs in clinical data!")
        raise ValueError("Duplicate patients found in clinical data!")
    
    # ------ 2. Load Pathology Data ------
    log_output("\n=== Loading Pathology Data ===")
    with open('response_features_portal_tract.pkl', 'rb') as f:
        response_features = pickle.load(f)
    
    with open('no_response_features_portal_tract.pkl', 'rb') as f:
        no_response_features = pickle.load(f)
    
    # Combine features
    X_pathology = np.vstack([
        response_features['embeddings'],
        no_response_features['embeddings']
    ])
    
    y_pathology = np.concatenate([
        response_features['labels'],
        no_response_features['labels']
    ])
    
    all_paths = response_features['paths'] + no_response_features['paths']
    
    # Extract patient IDs
    patient_ids_pathology = []
    for path in all_paths:
        patient_id = extract_patient_id_from_path(path)
        patient_ids_pathology.append(patient_id)
    
    log_output(f"Pathology data shape: {X_pathology.shape}")
    log_output(f"Response patches: {len(response_features['embeddings'])}")
    log_output(f"No Response patches: {len(no_response_features['embeddings'])}")
    
    # ------ 3. Aggregate Pathology Features ------
    log_output("\n=== Aggregating Pathology Features ===")
    all_aggregated_pathology = aggregate_pathology_features(
        all_paths, X_pathology, patient_ids_pathology, y_pathology, min_patches=1
    )
    
    log_output(f"Aggregated pathology data for {len(all_aggregated_pathology)} patients")
    
    # ------ 4. Create Multimodal Dataset ------
    log_output("\n=== Creating Multimodal Dataset ===")
    multimodal_data = create_multimodal_dataset(
        X_clinical, all_aggregated_pathology, patient_ids_clinical, y_clinical
    )
    
    log_output(f"Multimodal dataset: {len(multimodal_data['patients'])} patients")
    log_output(f"Clinical features shape: {multimodal_data['clinical_features'].shape}")
    log_output(f"Pathology features shape: {multimodal_data['pathology_features'].shape}")
    
    # Check label distribution
    label_counts = Counter(multimodal_data['labels'])
    log_output(f"Label distribution: {dict(label_counts)}")
    
    # ------ NEW: Add Feature Visualization ------
    # After creating multimodal_data
    visualize_patient_embeddings(
        multimodal_data['clinical_features'], 
        multimodal_data['pathology_features'],
        multimodal_data['labels'],
        multimodal_data['patients']
    )
    exit(0)
    
    # ------ 5. Prepare Data for Evaluation ------
    X_clinical_all = multimodal_data['clinical_features']
    X_pathology_all = multimodal_data['pathology_features']
    X_combined_all = np.concatenate((X_clinical_all, X_pathology_all), axis=1)
    y_all = multimodal_data['labels']
    patient_ids = multimodal_data['patients']
    
    log_output(f"Combined feature dimension: {X_combined_all.shape[1]}")
    log_output(f"Clinical: {X_clinical_all.shape[1]}, Pathology: {X_pathology_all.shape[1]}")
    
    # ------ 6. Get Model Configurations ------
    param_grids, fusion_param_grids = get_hyperparameter_grids()
    base_models = create_model_instances()
    fusion_base_models = create_fusion_model_instances()
    
    # Use 5-fold CV for outer loop
    outer_cv_folds = 5
    # Use 5-fold CV for inner hyperparameter tuning
    inner_cv_folds = 5
    log_output(f"Using {outer_cv_folds}-fold outer CV with {inner_cv_folds}-fold inner CV for hyperparameter tuning")
    
    # ------ 7. Create Data Structure for Results ------
    all_results = {
        "Clinical": {model_name: {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'f1_weighted': [], 'best_params': []} 
                    for model_name in base_models.keys()},
        "Pathology": {model_name: {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'f1_weighted': [], 'best_params': []} 
                     for model_name in base_models.keys()},
        "Fusion": {model_name: {'accuracy': [], 'balanced_accuracy': [], 'auc': [], 'f1_weighted': [], 'best_params': []} 
                  for model_name in fusion_base_models.keys()}
    }
    
    # Get unique patients for patient-based CV
    unique_patients = np.array(list(set(patient_ids)))
    
    # Create patient-based folds for the outer CV
    kf = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=SEED)
    
    # ------ 8. Perform Nested Cross Validation ------
    log_output("\n=== Starting Nested Cross-Validation ===")
    
    # Define model configurations
    model_configs = [
        {"name": "Clinical", "X": X_clinical_all, "models": base_models, "param_grids": param_grids},
        {"name": "Pathology", "X": X_pathology_all, "models": base_models, "param_grids": param_grids},
        {"name": "Fusion", "X": X_combined_all, "models": fusion_base_models, "param_grids": fusion_param_grids}
    ]
    
    # Outer CV loop
    for fold, (train_patient_idx, test_patient_idx) in enumerate(kf.split(unique_patients)):
        log_output(f"\n=== Outer Fold {fold+1}/{outer_cv_folds} ===")
        
        # Get patient IDs for this fold
        train_patients = set(unique_patients[train_patient_idx])
        test_patients = set(unique_patients[test_patient_idx])
        
        # Convert patient-based splits to sample indices
        train_indices = [i for i, p in enumerate(patient_ids) if p in train_patients]
        test_indices = [i for i, p in enumerate(patient_ids) if p in test_patients]
        
        y_train, y_test = y_all[train_indices], y_all[test_indices]
        
        # Skip this fold if there aren't enough classes in train or test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            log_output(f"⚠️ Fold {fold+1}: Skipping due to insufficient class representation")
            continue
        
        log_output(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
        log_output(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        log_output(f"Train label distribution: {dict(Counter(y_train))}")
        log_output(f"Test label distribution: {dict(Counter(y_test))}")
        
        # Process each model configuration
        for config in model_configs:
            model_type = config["name"]
            X = config["X"]
            X_train, X_test = X[train_indices], X[test_indices]
            
            log_output(f"\n--- Processing {model_type} Models for Fold {fold+1} ---")
            
            # For each model type
            for model_name, base_model in config["models"].items():
                log_output(f"  {model_type} {model_name.upper()} - Tuning hyperparameters...")
                
                # Get parameter grid for this model
                param_grid = config["param_grids"][model_name]
                
                # Tune hyperparameters on training data only with inner CV
                best_model, best_params, best_score = tune_hyperparameters_on_training_set(
                    X_train, y_train, model_name, base_model, param_grid, inner_cv=inner_cv_folds
                )
                
                if best_model is None:
                    log_output(f"  ❌ Failed to tune {model_type} {model_name.upper()}")
                    continue
                
                # Extract hyperparameters for display
                param_display = {}
                if isinstance(best_model, Pipeline):
                    # For pipeline models
                    for key, value in best_params.items():
                        if key.startswith('classifier__'):
                            clean_key = key.replace('classifier__', '')
                            param_display[clean_key] = value
                else:
                    # For direct models
                    param_display = best_params
                
                log_output(f"  Best params: {param_display}")
                log_output(f"  Best inner CV score: {best_score:.4f}")
                
                # Train best model on all training data
                best_model.fit(X_train, y_train)
                
                # Get predictions on test data
                y_pred = best_model.predict(X_test)
                
                # Get probabilities if available
                try:
                    if hasattr(best_model, 'predict_proba'):
                        y_proba = best_model.predict_proba(X_test)
                        if y_proba.shape[1] == 2:  # Binary classification
                            y_proba = y_proba[:, 1]
                    else:
                        y_proba = None
                except:
                    y_proba = None
                
                # Calculate metrics
                metrics = evaluate_model_with_metrics(y_test, y_pred, y_proba)
                
                # Store results
                all_results[model_type][model_name]['accuracy'].append(metrics['accuracy'])
                all_results[model_type][model_name]['balanced_accuracy'].append(metrics['balanced_accuracy'])
                all_results[model_type][model_name]['f1_weighted'].append(metrics['f1_weighted'])
                all_results[model_type][model_name]['best_params'].append(param_display)
                
                if metrics['auc'] is not None:
                    all_results[model_type][model_name]['auc'].append(metrics['auc'])
                
                # Log metrics for this fold
                log_output(f"  {model_type} {model_name.upper()} - Fold {fold+1} Results:")
                log_output(f"    Accuracy: {metrics['accuracy']:.4f}")
                log_output(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                log_output(f"    F1 Weighted: {metrics['f1_weighted']:.4f}")
                if metrics['auc'] is not None:
                    log_output(f"    AUC: {metrics['auc']:.4f}")
                else:
                    log_output(f"    AUC: N/A")
    
    # ------ 9. Calculate Summary Statistics ------
    log_output("\n=== Final Results Summary ===")
    
    # Prepare data for CSV
    csv_data = []
    
    for model_type in all_results.keys():
        for model_name, metrics in all_results[model_type].items():
            # Calculate mean and std for each metric
            acc_mean = np.mean(metrics['accuracy']) if metrics['accuracy'] else np.nan
            acc_std = np.std(metrics['accuracy']) if metrics['accuracy'] else np.nan
            bal_acc_mean = np.mean(metrics['balanced_accuracy']) if metrics['balanced_accuracy'] else np.nan
            bal_acc_std = np.std(metrics['balanced_accuracy']) if metrics['balanced_accuracy'] else np.nan
            f1_w_mean = np.mean(metrics['f1_weighted']) if metrics['f1_weighted'] else np.nan
            f1_w_std = np.std(metrics['f1_weighted']) if metrics['f1_weighted'] else np.nan
            
            # Handle AUC separately since it might be None for some folds
            auc_values = [a for a in metrics['auc'] if a is not None]
            auc_mean = np.mean(auc_values) if auc_values else np.nan
            auc_std = np.std(auc_values) if auc_values else np.nan
            
            # Count how many times each parameter value was selected
            param_counts = defaultdict(lambda: defaultdict(int))
            for params in metrics['best_params']:
                for param_name, param_value in params.items():
                    param_counts[param_name][param_value] += 1
            
            # Format most frequently selected parameters
            top_params = {}
            for param_name, value_counts in param_counts.items():
                top_value = max(value_counts.items(), key=lambda x: x[1])[0]
                top_count = value_counts[top_value]
                total = sum(value_counts.values())
                top_params[param_name] = f"{top_value} ({top_count}/{total})"
            
            param_str = "; ".join([f"{k}={v}" for k, v in top_params.items()])
            
            # Log results
            log_output(f"\n{model_type} {model_name.upper()} Overall Results:")
            log_output(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
            log_output(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
            log_output(f"  F1 Weighted: {f1_w_mean:.4f} ± {f1_w_std:.4f}")
            log_output(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}" if not np.isnan(auc_mean) else "  AUC: N/A")
            log_output(f"  Most frequent hyperparameters: {param_str}")
            
            # Add to CSV data
            csv_data.append({
                'model_type': model_type,
                'model_name': model_name.upper(),
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'balanced_accuracy_mean': bal_acc_mean,
                'balanced_accuracy_std': bal_acc_std,
                'f1_weighted_mean': f1_w_mean,
                'f1_weighted_std': f1_w_std,
                'auc_mean': auc_mean if not np.isnan(auc_mean) else 0.0,
                'auc_std': auc_std if not np.isnan(auc_std) else 0.0,
                'most_frequent_params': param_str
            })
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(csv_data)
    
    # Sort by balanced accuracy (primary metric)
    results_df = results_df.sort_values('balanced_accuracy_mean', ascending=False)
    
    # Display top results
    log_output("\n=== Top Models by Balanced Accuracy ===")
    for i, row in results_df.iterrows():
        auc_display = f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}" if row['auc_mean'] > 0 else "N/A"
        log_output(f"{i+1}. {row['model_type']} {row['model_name']}: "
                  f"Bal_Acc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}, "
                  f"Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
                  f"F1={row['f1_weighted_mean']:.4f}±{row['f1_weighted_std']:.4f}, "
                  f"AUC={auc_display}")
    
    # Find best in each category
    log_output("\n=== Best Models by Category (Balanced Accuracy) ===")
    for model_type in ['Clinical', 'Pathology', 'Fusion']:
        subset = results_df[results_df['model_type'] == model_type]
        if not subset.empty:
            best = subset.iloc[0]
            log_output(f"Best {model_type}: {best['model_name']} - "
                      f"Bal_Acc={best['balanced_accuracy_mean']:.4f}±{best['balanced_accuracy_std']:.4f}")
    
    # Save results to CSV
    results_df.to_csv('nested_cv_results.csv', index=False)
    log_output("Results saved to nested_cv_results.csv")
    
    # Save detailed results
    detailed_results = {
        'raw_results': all_results,
        'dataset_info': {
            'n_patients': len(multimodal_data['patients']),
            'clinical_features': X_clinical_all.shape[1],
            'pathology_features': X_pathology_all.shape[1],
            'patient_ids': patient_ids
        }
    }
    
    with open('detailed_nested_cv_results.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)
    log_output("Detailed results saved to detailed_nested_cv_results.pkl")
    
    log_output("\n=== Analysis Complete ===")
    log_output("Summary:")
    log_output(f"✓ Performed nested {outer_cv_folds}-fold cross-validation with {inner_cv_folds}-fold inner CV")
    log_output(f"✓ Evaluated 4 Clinical models, 4 Pathology models, and 4 Fusion models")
    log_output(f"✓ Evaluated metrics: Accuracy, Balanced Accuracy, F1 Weighted, AUROC")
    if not results_df.empty:
        log_output(f"✓ Best overall model: {results_df.iloc[0]['model_type']} {results_df.iloc[0]['model_name']}")
        log_output(f"  Balanced Accuracy: {results_df.iloc[0]['balanced_accuracy_mean']:.4f}±{results_df.iloc[0]['balanced_accuracy_std']:.4f}")
    else:
        log_output("❌ No successful model results found")


if __name__ == "__main__":
    main()