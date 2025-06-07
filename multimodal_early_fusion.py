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


def perform_hyperparameter_tuning(X, y, model_type, models, param_grids):
    """Perform hyperparameter tuning on the whole dataset and return best models based on accuracy."""
    best_models = {}
    
    log_output(f"\n=== Hyperparameter Tuning for {model_type} Models ===")
    
    for name, model in models.items():
        log_output(f"Tuning {model_type} {name.upper()}...")
        
        # Create pipeline with scaling for models that need it
        if name in ['lr', 'svm']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            # Adjust parameter names for pipeline
            param_grid = {}
            for key, value in param_grids[name].items():
                param_grid[f'classifier__{key}'] = value
        else:
            pipeline = model
            param_grid = param_grids[name]
        
        try:
            # Hyperparameter tuning on whole dataset
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5,  # Use 5-fold CV for parameter selection
                scoring='accuracy',  # Use accuracy for tuning as requested
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            
            log_output(f"Best parameters for {model_type} {name.upper()}: {grid_search.best_params_}")
            log_output(f"Best accuracy score: {grid_search.best_score_:.4f}")
            
            best_models[name] = best_model
            
        except Exception as e:
            log_output(f"❌ Error tuning {model_type} {name}: {e}")
    
    return best_models


def main():
    log_output("=== Starting Hyperparameter Tuning and Patient-based Fold Evaluation for Multimodal Early Fusion ===")
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
    models = create_model_instances()
    fusion_models = create_fusion_model_instances()
    
    # Use 5-fold CV
    cv_folds = 5
    log_output(f"Using {cv_folds}-fold cross-validation")
    
    # ------ 7. Hyperparameter Tuning for All Models on Whole Dataset ------
    # Clinical models
    clinical_best_models = perform_hyperparameter_tuning(
        X_clinical_all, y_all, "Clinical", models, param_grids
    )
    
    # Pathology models
    pathology_best_models = perform_hyperparameter_tuning(
        X_pathology_all, y_all, "Pathology", models, param_grids
    )
    
    # Fusion models
    fusion_best_models = perform_hyperparameter_tuning(
        X_combined_all, y_all, "Fusion", fusion_models, fusion_param_grids
    )
    
    # ------ 8. Patient-based Fold Evaluation ------
    log_output("\n=== Patient-based Fold Evaluation ===")
    
    # Create dictionary to hold all models and their data
    all_model_configs = [
        {"name": "Clinical", "X": X_clinical_all, "models": clinical_best_models},
        {"name": "Pathology", "X": X_pathology_all, "models": pathology_best_models},
        {"name": "Fusion", "X": X_combined_all, "models": fusion_best_models}
    ]
    
    # Initialize results dictionary for all models
    all_results = {}
    for config in all_model_configs:
        model_type = config["name"]
        all_results[model_type] = {}
        for model_name in config["models"].keys():
            all_results[model_type][model_name] = {
                'accuracy': [],
                'balanced_accuracy': [],
                'auc': [],
                'f1_weighted': [],
                'params': config["models"][model_name].get_params()
            }
    
    # Get unique patients
    unique_patients = np.array(list(set(patient_ids)))
    
    # Create patient-based folds
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    # Evaluate each fold
    for fold, (train_patient_idx, test_patient_idx) in enumerate(kf.split(unique_patients)):
        log_output(f"\n--- Evaluating Fold {fold+1}/{cv_folds} ---")
        
        train_patients = set(unique_patients[train_patient_idx])
        test_patients = set(unique_patients[test_patient_idx])
        
        # Convert patient-based splits to sample indices
        train_indices = [i for i, p in enumerate(patient_ids) if p in train_patients]
        test_indices = [i for i, p in enumerate(patient_ids) if p in test_patients]
        
        y_train, y_test = y_all[train_indices], y_all[test_indices]
        
        # Skip this fold if there aren't enough classes
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            log_output(f"⚠️ Fold {fold+1}: Skipping due to insufficient class representation")
            continue
        
        log_output(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
        log_output(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        
        # Evaluate each model configuration
        for config in all_model_configs:
            model_type = config["name"]
            X = config["X"]
            X_train, X_test = X[train_indices], X[test_indices]
            
            log_output(f"\n--- Evaluating {model_type} Models for Fold {fold+1} ---")
            
            # Evaluate each model of this type
            for model_name, model in config["models"].items():
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model_clone.predict(X_test)
                
                # Get probabilities if available
                try:
                    y_proba = model_clone.predict_proba(X_test)[:, 1] if hasattr(model_clone, 'predict_proba') else None
                except:
                    y_proba = None
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1_w = f1_score(y_test, y_pred, average='weighted')
                
                auc = None
                if y_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                    except:
                        auc = None
                
                # Store metrics
                all_results[model_type][model_name]['accuracy'].append(acc)
                all_results[model_type][model_name]['balanced_accuracy'].append(bal_acc)
                all_results[model_type][model_name]['f1_weighted'].append(f1_w)
                if auc is not None:
                    all_results[model_type][model_name]['auc'].append(auc)
                
                # Log results
                log_output(f"{model_type} {model_name.upper()} - Fold {fold+1}:")
                log_output(f"  Accuracy: {acc:.4f}")
                log_output(f"  Balanced Accuracy: {bal_acc:.4f}")
                log_output(f"  F1 Weighted: {f1_w:.4f}")
                log_output(f"  AUC: {auc:.4f}" if auc is not None else "  AUC: N/A")
    
    # ------ 9. Calculate mean and std for metrics across all folds ------
    log_output("\n=== Final Results Summary ===")
    
    # Create dictionary to map model_type to best models dictionary
    model_type_to_best_models = {
        'Clinical': clinical_best_models,
        'Pathology': pathology_best_models,
        'Fusion': fusion_best_models
    }
    
    # Prepare data for CSV
    csv_data = []
    
    for model_type, models_dict in all_results.items():
        for model_name, metrics in models_dict.items():
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
            
            # Extract best hyperparameters
            best_model = model_type_to_best_models[model_type].get(model_name)
            best_params = {}
            
            if best_model is not None:
                # Handle pipeline vs. direct model
                if hasattr(best_model, 'get_params'):
                    all_params = best_model.get_params()
                    
                    if isinstance(best_model, Pipeline):
                        # For pipeline models, extract classifier params
                        classifier_params = {k.replace('classifier__', ''): v 
                                            for k, v in all_params.items() 
                                            if k.startswith('classifier__')}
                        best_params = classifier_params
                    else:
                        # For non-pipeline models, get only the relevant hyperparameters
                        relevant_params = {}
                        for param_name in param_grids.get(model_name, {}).keys():
                            if param_name in all_params:
                                relevant_params[param_name] = all_params[param_name]
                        best_params = relevant_params
            
            # Format hyperparameters as string
            param_str = "; ".join([f"{k}={v}" for k, v in best_params.items() if not callable(v)])
            
            # Log results
            log_output(f"\n{model_type} {model_name.upper()} Overall Results:")
            log_output(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
            log_output(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
            log_output(f"  F1 Weighted: {f1_w_mean:.4f} ± {f1_w_std:.4f}")
            log_output(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}" if not np.isnan(auc_mean) else "  AUC: N/A")
            log_output(f"  Best hyperparameters: {param_str}")
            
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
                'best_hyperparameters': param_str
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
    results_df.to_csv('patient_fold_evaluation_results.csv', index=False)
    log_output("Results saved to patient_fold_evaluation_results.csv")
    
    # Save detailed results
    detailed_results = {
        'raw_results': all_results,
        'clinical_best_models': clinical_best_models,
        'pathology_best_models': pathology_best_models,
        'fusion_best_models': fusion_best_models,
        'dataset_info': {
            'n_patients': len(multimodal_data['patients']),
            'clinical_features': X_clinical_all.shape[1],
            'pathology_features': X_pathology_all.shape[1],
            'patient_ids': patient_ids
        }
    }
    
    with open('detailed_patient_fold_results.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)
    log_output("Detailed results saved to detailed_patient_fold_results.pkl")
    
    log_output("\n=== Analysis Complete ===")
    log_output("Summary:")
    log_output(f"✓ Tuned and evaluated 4 Clinical models with {cv_folds} patient-based folds")
    log_output(f"✓ Tuned and evaluated 4 Pathology models with {cv_folds} patient-based folds") 
    log_output(f"✓ Tuned and evaluated 4 Fusion models with {cv_folds} patient-based folds")
    log_output(f"✓ Evaluated metrics: Accuracy, Balanced Accuracy, F1 Weighted, AUROC")
    if not results_df.empty:
        log_output(f"✓ Best overall model: {results_df.iloc[0]['model_type']} {results_df.iloc[0]['model_name']}")
        log_output(f"  Balanced Accuracy: {results_df.iloc[0]['balanced_accuracy_mean']:.4f}±{results_df.iloc[0]['balanced_accuracy_std']:.4f}")
    else:
        log_output("❌ No successful model results found")


if __name__ == "__main__":
    main()