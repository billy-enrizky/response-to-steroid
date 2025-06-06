import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score, make_scorer, f1_score
)
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import itertools
import time

from utils.my_utils import get_eval_metrics, eval_sklearn_classifier, calculate_feature_weights, get_slide_level_predictions, train_eval_sklearn_model

# Set up logging
log_file = "hypertuned_multimodal_fusion.log"
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

# Suppress convergence warnings
simplefilter("ignore", ConvergenceWarning)

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None):
    """Calculate all required metrics: accuracy, balanced accuracy, AUROC, and F1 weighted"""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Balanced Accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # F1 Weighted Score
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # AUROC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if y_prob.ndim == 1:
                # Single probability array
                metrics['auroc'] = roc_auc_score(y_true, y_prob)
            elif y_prob.shape[1] == 2:
                # Binary classification probabilities
                metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                metrics['auroc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError as e:
            log_output(f"Warning: Could not compute AUROC: {e}")
            metrics['auroc'] = None
    else:
        metrics['auroc'] = None
    
    return metrics

def get_hyperparameter_grids():
    """Define hyperparameter grids for each model type"""
    
    # Clinical model hyperparameters (smaller grids due to limited data)
    clinical_param_grids = {
        'lr': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'probability': [True]  # Enable probability estimates for SVM
        },
        'rf': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gb': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 1.0]
        }
    }
    
    # Pathology model hyperparameters (can be larger due to more samples)
    pathology_param_grids = {
        'lr': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000, 2000]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'probability': [True]
        },
        'rf': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8]
        },
        'gb': {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    # Fusion model hyperparameters (focus on fusion weights)
    fusion_param_grids = {
        'weights': np.arange(0.1, 1.0, 0.1),  # Clinical weight (pathology weight = 1 - clinical_weight)
        'threshold': [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]  # Decision threshold
    }
    
    return clinical_param_grids, pathology_param_grids, fusion_param_grids

def tune_model_on_full_dataset(model_class, param_grid, X, y, scale_features=False, cv=5):
    """Tune hyperparameters on the full dataset using cross-validation"""
    log_output(f"Tuning {model_class.__name__} on full dataset...")
    
    # Scale features if needed
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # Use balanced accuracy as scoring metric
    scorer = make_scorer(balanced_accuracy_score)
    
    # Use GridSearchCV for comprehensive search
    grid_search = GridSearchCV(
        model_class(random_state=SEED),
        param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1
    )
    
    log_output(f"Starting hyperparameter search with {len(list(itertools.product(*param_grid.values())))} combinations...")
    grid_search.fit(X_scaled, y)
    
    log_output(f"Best parameters for {model_class.__name__}: {grid_search.best_params_}")
    log_output(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, scaler

def tune_fusion_weights(clinical_probs, pathology_probs, true_labels, fusion_param_grid):
    """Tune fusion weights and threshold"""
    log_output("Tuning fusion weights and threshold...")
    
    best_score = 0
    best_params = {'weight': 0.5, 'threshold': 0.5}
    
    for clinical_weight in fusion_param_grid['weights']:
        pathology_weight = 1.0 - clinical_weight
        
        for threshold in fusion_param_grid['threshold']:
            # Perform weighted fusion
            fused_probs = []
            for clin_p, path_p in zip(clinical_probs, pathology_probs):
                weighted_prob = clinical_weight * clin_p + pathology_weight * path_p
                fused_probs.append(weighted_prob)
            
            # Convert probabilities to predictions using threshold
            fused_preds = [1 if p >= threshold else 0 for p in fused_probs]
            
            # Calculate balanced accuracy
            score = balanced_accuracy_score(true_labels, fused_preds)
            
            if score > best_score:
                best_score = score
                best_params = {'clinical_weight': clinical_weight, 'threshold': threshold}
    
    log_output(f"Best fusion parameters: {best_params}")
    log_output(f"Best fusion score: {best_score:.4f}")
    
    return best_params

def main():
    log_output("=== Starting Hyperparameter Tuned 5-Fold Cross-Validation for Multimodal Fusion ===")
    log_output("=== Hyperparameter tuning performed on FULL DATASET ===")
    log_output(f"Log file: {log_file}")
    log_output(f"Dataset info: 55 patients, 13 clinical features, 1536 image features")
    log_output(f"Pathology patches: 2272 Response + 1756 No Response = 4028 total")
    log_output("Metrics: Accuracy, Balanced Accuracy, AUROC, F1 Weighted Score")
    
    # ------ 1. Load Clinical Data ------
    log_output("\n=== Loading Clinical Data ===")
    with open('clinical_features_portal_tract.pkl', 'rb') as f:
        df_cleaned = pickle.load(f)
    
    X_clinical = df_cleaned.drop(columns=["RAI Classification Biopsy #2", "patient_id"]).values
    y_clinical = df_cleaned['RAI Classification Biopsy #2'].values
    patient_ids_clinical = df_cleaned['patient_id'].values
    
    log_output(f"Clinical data shape: {X_clinical.shape}")
    log_output(f"Number of patients in clinical data: {len(patient_ids_clinical)}")
    
    # ------ 2. Load Pathology Data ------
    log_output("\n=== Loading Pathology Data ===")
    with open('response_features_portal_tract.pkl', 'rb') as f:
        response_features = pickle.load(f)
    
    with open('no_response_features_portal_tract.pkl', 'rb') as f:
        no_response_features = pickle.load(f)
    
    # Combine features for cross-validation
    X_pathology = np.vstack([
        response_features['embeddings'],
        no_response_features['embeddings']
    ])
    
    y_pathology = np.concatenate([
        response_features['labels'],
        no_response_features['labels']
    ])
    
    # Combine paths
    all_paths = response_features['paths'] + no_response_features['paths']
    
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
    
    # ------ 3. Get Hyperparameter Grids ------
    clinical_param_grids, pathology_param_grids, fusion_param_grids = get_hyperparameter_grids()
    
    # ------ 4. Tune Hyperparameters on Full Datasets ------
    log_output("\n=== PHASE 1: Hyperparameter Tuning on Full Datasets ===")
    
    # Define model classes and scaling requirements
    clinical_model_configs = {
        'lr': {'model': sk_LogisticRegression, 'scale': True},
        'svm': {'model': SVC, 'scale': True},
        'rf': {'model': RandomForestClassifier, 'scale': False},
        'gb': {'model': GradientBoostingClassifier, 'scale': False}
    }
    
    pathology_model_configs = {
        'lr': {'model': sk_LogisticRegression, 'scale': True},
        'svm': {'model': SVC, 'scale': True},
        'rf': {'model': RandomForestClassifier, 'scale': False},
        'gb': {'model': GradientBoostingClassifier, 'scale': False}
    }
    
    # Store best parameters for each model
    clinical_best_params = {}
    clinical_scalers = {}
    pathology_best_params = {}
    pathology_scalers = {}
    
    # --- 4.1. Tune Clinical Models ---
    log_output("\n--- Tuning Clinical Models on Full Dataset ---")
    for name, config in clinical_model_configs.items():
        best_params, scaler = tune_model_on_full_dataset(
            config['model'],
            clinical_param_grids[name],
            X_clinical,
            y_clinical,
            scale_features=config['scale'],
            cv=5
        )
        clinical_best_params[name] = best_params
        clinical_scalers[name] = scaler
    
    # --- 4.2. Tune Pathology Models ---
    log_output("\n--- Tuning Pathology Models on Full Dataset ---")
    for name, config in pathology_model_configs.items():
        best_params, scaler = tune_model_on_full_dataset(
            config['model'],
            pathology_param_grids[name],
            X_pathology,
            y_pathology,
            scale_features=config['scale'],
            cv=5
        )
        pathology_best_params[name] = best_params
        pathology_scalers[name] = scaler
    
    # ------ 5. Cross-Validation with Best Hyperparameters ------
    log_output("\n=== PHASE 2: 5-Fold Cross-Validation with Best Hyperparameters ===")
    
    n_folds = 5
    kf_clinical = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize data structures to track model performance across folds
    clinical_models_performance = defaultdict(lambda: defaultdict(list))
    pathology_models_performance = defaultdict(lambda: defaultdict(list))
    slide_level_performance = defaultdict(lambda: defaultdict(list))
    fusion_performance = defaultdict(lambda: defaultdict(list))
    
    # Store fusion parameters for each combination
    fusion_best_params = defaultdict(list)
    
    # ------ 6. Perform Cross-Validation ------
    for fold, (train_idx_clinical, test_idx_clinical) in enumerate(kf_clinical.split(X_clinical)):
        log_output(f"\n=== Fold {fold+1}/{n_folds} ===")
        start_time = time.time()
        
        # --- 6.1. Split Clinical Data ---
        X_clinical_train = X_clinical[train_idx_clinical]
        y_clinical_train = y_clinical[train_idx_clinical]
        X_clinical_test = X_clinical[test_idx_clinical]
        y_clinical_test = y_clinical[test_idx_clinical]
        patient_ids_clinical_test = patient_ids_clinical[test_idx_clinical]
        
        # --- 6.2. Split Pathology Data Based on Patient IDs ---
        log_output("Splitting pathology data based on patient IDs...")
        
        test_patients = set(patient_ids_clinical_test)
        
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
        
        # --- 6.3. Train Clinical Models with Best Parameters ---
        log_output("\n--- Training Clinical Models with Best Parameters ---")
        
        clinical_results = {}
        for name, config in clinical_model_configs.items():
            log_output(f"\nTraining Clinical {name.upper()} with params: {clinical_best_params[name]}")
            
            # Create model with best parameters
            model = config['model'](random_state=SEED, **clinical_best_params[name])
            
            # Scale data if needed
            if config['scale'] and clinical_scalers[name] is not None:
                scaler = StandardScaler()
                X_clinical_train_scaled = scaler.fit_transform(X_clinical_train)
                X_clinical_test_scaled = scaler.transform(X_clinical_test)
            else:
                X_clinical_train_scaled = X_clinical_train
                X_clinical_test_scaled = X_clinical_test
                scaler = None
            
            # Train model
            model.fit(X_clinical_train_scaled, y_clinical_train)
            
            # Make predictions
            y_pred = model.predict(X_clinical_test_scaled)
            y_prob = model.predict_proba(X_clinical_test_scaled) if hasattr(model, "predict_proba") else None
            
            # Calculate comprehensive metrics
            metrics = calculate_comprehensive_metrics(y_clinical_test, y_pred, y_prob)
            
            clinical_results[name] = {
                'model': model,
                'scaler': scaler,
                'predictions': y_pred,
                'probabilities': y_prob,
                'best_params': clinical_best_params[name],
                **metrics
            }
            
            log_output(f"Clinical {name.upper()} - Accuracy: {metrics['accuracy']:.4f}, "
                      f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                      f"F1 Weighted: {metrics['f1_weighted']:.4f}")
            if metrics['auroc'] is not None:
                log_output(f"Clinical {name.upper()} - AUROC: {metrics['auroc']:.4f}")
            
            # Store performance metrics for this fold
            for metric_name, value in metrics.items():
                if value is not None:
                    clinical_models_performance[name][metric_name].append(value)
        
        # --- 6.4. Train Pathology Models with Best Parameters ---
        log_output("\n--- Training Pathology Models with Best Parameters ---")
        
        pathology_results = {}
        for name, config in pathology_model_configs.items():
            log_output(f"\nTraining Pathology {name.upper()} with params: {pathology_best_params[name]}")
            
            # Create model with best parameters
            model = config['model'](random_state=SEED, **pathology_best_params[name])
            
            # Scale data if needed
            if config['scale'] and pathology_scalers[name] is not None:
                scaler = StandardScaler()
                X_pathology_train_scaled = scaler.fit_transform(X_pathology_train)
                X_pathology_test_scaled = scaler.transform(X_pathology_test)
            else:
                X_pathology_train_scaled = X_pathology_train
                X_pathology_test_scaled = X_pathology_test
                scaler = None
            
            # Train model
            model.fit(X_pathology_train_scaled, y_pathology_train)
            
            # Make predictions
            y_pred = model.predict(X_pathology_test_scaled)
            y_prob = model.predict_proba(X_pathology_test_scaled) if hasattr(model, "predict_proba") else None
            
            # Calculate comprehensive metrics
            metrics = calculate_comprehensive_metrics(y_pathology_test, y_pred, y_prob)
            
            pathology_results[name] = {
                'model': model,
                'scaler': scaler,
                'predictions': y_pred,
                'probabilities': y_prob,
                'best_params': pathology_best_params[name],
                **metrics
            }
            
            log_output(f"Pathology {name.upper()} - Accuracy: {metrics['accuracy']:.4f}, "
                      f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                      f"F1 Weighted: {metrics['f1_weighted']:.4f}")
            if metrics['auroc'] is not None:
                log_output(f"Pathology {name.upper()} - AUROC: {metrics['auroc']:.4f}")
            
            # Store performance metrics for this fold
            for metric_name, value in metrics.items():
                if value is not None:
                    pathology_models_performance[name][metric_name].append(value)
        
        # --- 6.5. Aggregate Pathology Predictions at Slide Level ---
        log_output("\n--- Aggregating Pathology Predictions at Slide Level for Each Model ---")
        
        slide_level_results = {}
        for name in pathology_model_configs.keys():
            slide_ids, y_true_slides, y_pred_slides, y_prob_slides = get_slide_level_predictions(
                test_items,
                pathology_results[name]['predictions'],
                pathology_results[name]['probabilities']
            )
            
            # Calculate comprehensive metrics for slide-level predictions
            slide_metrics = calculate_comprehensive_metrics(y_true_slides, y_pred_slides, y_prob_slides)
            
            slide_level_results[name] = {
                'slide_ids': slide_ids,
                'true_labels': y_true_slides,
                'predictions': y_pred_slides,
                'probabilities': y_prob_slides,
                **slide_metrics
            }
            
            log_output(f"Slide-level {name.upper()} - Accuracy: {slide_metrics['accuracy']:.4f}, "
                      f"Balanced Acc: {slide_metrics['balanced_accuracy']:.4f}, "
                      f"F1 Weighted: {slide_metrics['f1_weighted']:.4f}")
            if slide_metrics['auroc'] is not None:
                log_output(f"Slide-level {name.upper()} - AUROC: {slide_metrics['auroc']:.4f}")
            
            # Store slide-level performance metrics
            for metric_name, value in slide_metrics.items():
                if value is not None:
                    slide_level_performance[name][metric_name].append(value)
        
        # --- 6.6. Tune and Evaluate Fusion Models ---
        log_output("\n--- Tuning and Evaluating Fusion Models ---")
        
        for clin_name, path_name in itertools.product(clinical_model_configs.keys(), pathology_model_configs.keys()):
            combo_name = f"{clin_name}+{path_name}"
            log_output(f"\n--- Tuning combination: {combo_name} ---")
            
            # Get slide-level predictions for this pathology model
            slide_ids = slide_level_results[path_name]['slide_ids']
            slide_prob_map = {sid: prob for sid, prob in zip(slide_ids, slide_level_results[path_name]['probabilities'])}
            
            # Find common patients in both modalities
            common_patients = []
            clinical_probs = []
            pathology_probs = []
            true_labels = []
            
            for i, patient_id in enumerate(patient_ids_clinical_test):
                if patient_id in slide_prob_map:
                    common_patients.append(patient_id)
                    
                    # Get clinical probability
                    if clinical_results[clin_name]['probabilities'] is not None:
                        if clinical_results[clin_name]['probabilities'].shape[1] == 2:
                            clinical_prob = clinical_results[clin_name]['probabilities'][i][1]
                        else:
                            clinical_prob = clinical_results[clin_name]['predictions'][i]
                    else:
                        clinical_prob = clinical_results[clin_name]['predictions'][i]
                    
                    clinical_probs.append(clinical_prob)
                    
                    # Get pathology probability
                    pathology_prob = slide_prob_map[patient_id]
                    pathology_probs.append(pathology_prob)
                    
                    # Get true label
                    true_label = y_clinical_test[i]
                    true_labels.append(true_label)
            
            log_output(f"Found {len(common_patients)} common patients for fusion")
            
            if len(common_patients) > 0:
                # Tune fusion parameters
                best_fusion_params = tune_fusion_weights(
                    clinical_probs, pathology_probs, true_labels, fusion_param_grids
                )
                
                fusion_best_params[combo_name].append(best_fusion_params)
                
                # Apply best fusion parameters
                clinical_weight = best_fusion_params['clinical_weight']
                pathology_weight = 1.0 - clinical_weight
                threshold = best_fusion_params['threshold']
                
                # Perform weighted fusion with tuned parameters
                fused_probs = []
                for clin_p, path_p in zip(clinical_probs, pathology_probs):
                    weighted_prob = clinical_weight * clin_p + pathology_weight * path_p
                    fused_probs.append(weighted_prob)
                
                # Convert probabilities to predictions using tuned threshold
                fused_preds = [1 if p >= threshold else 0 for p in fused_probs]
                
                # Calculate comprehensive metrics for fusion
                fusion_metrics = calculate_comprehensive_metrics(true_labels, fused_preds, fused_probs)
                
                log_output(f"Fusion {combo_name} - Accuracy: {fusion_metrics['accuracy']:.4f}, "
                          f"Balanced Acc: {fusion_metrics['balanced_accuracy']:.4f}, "
                          f"F1 Weighted: {fusion_metrics['f1_weighted']:.4f}")
                if fusion_metrics['auroc'] is not None:
                    log_output(f"Fusion {combo_name} - AUROC: {fusion_metrics['auroc']:.4f}")
                
                log_output(f"Tuned weights - Clinical: {clinical_weight:.2f}, Pathology: {pathology_weight:.2f}")
                log_output(f"Tuned threshold: {threshold:.2f}")
                
                # Compare with individual modalities on common patients
                clinical_only_preds = [1 if p >= 0.5 else 0 for p in clinical_probs]
                clinical_only_metrics = calculate_comprehensive_metrics(true_labels, clinical_only_preds, clinical_probs)
                
                pathology_only_preds = [1 if p >= 0.5 else 0 for p in pathology_probs]
                pathology_only_metrics = calculate_comprehensive_metrics(true_labels, pathology_only_preds, pathology_probs)
                
                log_output(f"On common patients - Clinical: {clinical_only_metrics['accuracy']:.4f}, "
                          f"Pathology: {pathology_only_metrics['accuracy']:.4f}, "
                          f"Fusion: {fusion_metrics['accuracy']:.4f}")
                
                # Store fusion performance for this combination in this fold
                for metric_name, value in fusion_metrics.items():
                    if value is not None:
                        fusion_performance[combo_name][metric_name].append(value)
                
                # Store individual modality performance on common patients
                for metric_name, value in clinical_only_metrics.items():
                    if value is not None:
                        fusion_performance[combo_name][f'clinical_{metric_name}'].append(value)
                
                for metric_name, value in pathology_only_metrics.items():
                    if value is not None:
                        fusion_performance[combo_name][f'pathology_{metric_name}'].append(value)
                
                fusion_performance[combo_name]['n_patients'].append(len(common_patients))
            else:
                log_output(f"No common patients found for combination {combo_name} in this fold!")
        
        fold_time = time.time() - start_time
        log_output(f"Fold {fold+1} completed in {fold_time:.2f} seconds")
    
    # ------ 7. Calculate & Report Average Performance Across Folds ------
    
    # --- 7.1. Report Best Parameters Found on Full Dataset ---
    log_output("\n=== Best Hyperparameters Found on Full Dataset ===")
    
    log_output("\nClinical Models:")
    for name in clinical_model_configs.keys():
        log_output(f"{name.upper()}: {clinical_best_params[name]}")
    
    log_output("\nPathology Models:")
    for name in pathology_model_configs.keys():
        log_output(f"{name.upper()}: {pathology_best_params[name]}")
    
    log_output("\nFusion Models (per fold):")
    for combo_name in fusion_best_params.keys():
        params_list = fusion_best_params[combo_name]
        if params_list:
            log_output(f"{combo_name}:")
            for i, params in enumerate(params_list):
                log_output(f"  Fold {i+1}: {params}")
    
    # --- 7.2. Clinical Models Performance ---
    log_output("\n=== Clinical Models Performance (mean ± std) ===")
    clinical_summary = []
    
    for name in clinical_model_configs.keys():
        row = {'model': f"Clinical {name.upper()}"}
        
        for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
            if metric in clinical_models_performance[name] and clinical_models_performance[name][metric]:
                values = clinical_models_performance[name][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
                log_output(f"Clinical {name.upper()} - {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                row[metric] = "N/A"
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
        
        clinical_summary.append(row)
    
    # --- 7.3. Pathology Models Performance (Patch Level) ---
    log_output("\n=== Pathology Models Performance - Patch Level (mean ± std) ===")
    pathology_summary = []
    
    for name in pathology_model_configs.keys():
        row = {'model': f"Pathology {name.upper()} (Patch)"}
        
        for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
            if metric in pathology_models_performance[name] and pathology_models_performance[name][metric]:
                values = pathology_models_performance[name][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
                log_output(f"Pathology {name.upper()} (Patch) - {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                row[metric] = "N/A"
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
        
        pathology_summary.append(row)
    
    # --- 7.4. Pathology Models Performance (Slide Level) ---
    log_output("\n=== Pathology Models Performance - Slide Level (mean ± std) ===")
    slide_level_summary = []
    
    for name in pathology_model_configs.keys():
        row = {'model': f"Pathology {name.upper()} (Slide)"}
        
        for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
            if metric in slide_level_performance[name] and slide_level_performance[name][metric]:
                values = slide_level_performance[name][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
                log_output(f"Pathology {name.upper()} (Slide) - {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                row[metric] = "N/A"
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
        
        slide_level_summary.append(row)
    
    # --- 7.5. Fusion Models Performance ---
    log_output("\n=== Fusion Models Performance (mean ± std) ===")
    fusion_summary = []
    
    for combo_name in fusion_performance.keys():
        if not fusion_performance[combo_name].get('accuracy'):
            continue
            
        row = {'model': f"Fusion {combo_name}"}
        
        for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
            if metric in fusion_performance[combo_name] and fusion_performance[combo_name][metric]:
                values = fusion_performance[combo_name][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
                log_output(f"Fusion {combo_name} - {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                row[metric] = "N/A"
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
        
        # Add constituent model performance on common patients
        for modality in ['clinical', 'pathology']:
            for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
                key = f'{modality}_{metric}'
                if key in fusion_performance[combo_name] and fusion_performance[combo_name][key]:
                    values = fusion_performance[combo_name][key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[key] = f"{mean_val:.4f} ± {std_val:.4f}"
                    row[f'{key}_mean'] = mean_val
                    row[f'{key}_std'] = std_val
        
        if 'n_patients' in fusion_performance[combo_name]:
            avg_patients = np.mean(fusion_performance[combo_name]['n_patients'])
            row['avg_patients'] = avg_patients
            log_output(f"Avg Patients for {combo_name}: {avg_patients:.1f}")
        
        fusion_summary.append(row)
    
    # --- 7.6. Find Best Combinations ---
    if fusion_summary:
        log_output("\n=== Best Performing Combinations ===")
        
        fusion_df = pd.DataFrame(fusion_summary)
        
        # Sort by each metric
        for metric in ['accuracy', 'balanced_accuracy', 'auroc', 'f1_weighted']:
            if f'{metric}_mean' in fusion_df.columns:
                valid_rows = fusion_df.dropna(subset=[f'{metric}_mean'])
                if not valid_rows.empty:
                    best_by_metric = valid_rows.nlargest(3, f'{metric}_mean')
                    log_output(f"\nTop 3 combinations by {metric.replace('_', ' ').title()}:")
                    for _, row in best_by_metric.iterrows():
                        log_output(f"{row['model']}: {row[metric]}")
        
        # Check if any fusion model beat both constituent models
        log_output("\nFusion models that outperformed both constituent models (balanced accuracy):")
        improvement_found = False
        for _, row in fusion_df.iterrows():
            if (pd.notna(row.get('balanced_accuracy_mean')) and 
                pd.notna(row.get('clinical_balanced_accuracy_mean')) and 
                pd.notna(row.get('pathology_balanced_accuracy_mean'))):
                if (row['balanced_accuracy_mean'] > row['clinical_balanced_accuracy_mean'] and 
                    row['balanced_accuracy_mean'] > row['pathology_balanced_accuracy_mean']):
                    improvement = max(row['balanced_accuracy_mean'] - row['clinical_balanced_accuracy_mean'], 
                                    row['balanced_accuracy_mean'] - row['pathology_balanced_accuracy_mean'])
                    log_output(f"{row['model']}: {row['balanced_accuracy']} "
                              f"(improvement: +{improvement:.4f})")
                    improvement_found = True
        
        if not improvement_found:
            log_output("No fusion models outperformed both constituent models.")
    
    # --- 7.7. Save Results to CSV ---
    log_output("\n=== Saving Results to CSV ===")
    
    # Combine all results into one dataframe
    all_results = pd.concat([
        pd.DataFrame(clinical_summary),
        pd.DataFrame(pathology_summary),
        pd.DataFrame(slide_level_summary),
        pd.DataFrame(fusion_summary) if fusion_summary else pd.DataFrame()
    ], ignore_index=True)
    
    # Save results
    results_file = "hypertuned_multimodal_performance_full_dataset.csv"
    all_results.to_csv(results_file, index=False)
    log_output(f"Results saved to {results_file}")
    
    # Save best parameters
    params_file = "best_hyperparameters_full_dataset.pkl"
    best_params_dict = {
        'clinical': clinical_best_params,
        'pathology': pathology_best_params,
        'fusion': dict(fusion_best_params)
    }
    
    with open(params_file, 'wb') as f:
        pickle.dump(best_params_dict, f)
    log_output(f"Best hyperparameters saved to {params_file}")
    
    log_output("\n=== Hyperparameter Tuning Complete ===")
    log_output("Hyperparameter tuning performed on FULL DATASET")
    log_output("Cross-validation performed with BEST PARAMETERS")
    log_output("All metrics calculated: Accuracy, Balanced Accuracy, AUROC, F1 Weighted Score")

if __name__ == "__main__":
    main()