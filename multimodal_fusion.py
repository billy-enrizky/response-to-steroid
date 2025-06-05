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
import itertools

from utils.my_utils import get_eval_metrics, eval_sklearn_classifier, calculate_feature_weights, get_slide_level_predictions, train_eval_sklearn_model

# Set up logging
log_file = "all_combinations_multimodal_fusion.log"
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


def main():
    log_output("=== Starting 5-Fold Cross-Validation for Multimodal Fusion (All Combinations) ===")
    log_output(f"Log file: {log_file}")
    
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
    # Load features from files
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
    
    # ------ 3. Set Up K-Fold Cross-Validation ------
    n_folds = 5
    kf_clinical = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize data structures to track model performance across folds
    clinical_models_performance = defaultdict(lambda: defaultdict(list))
    pathology_models_performance = defaultdict(lambda: defaultdict(list))
    slide_level_performance = defaultdict(lambda: defaultdict(list))
    fusion_performance = defaultdict(lambda: defaultdict(list))
    
    # Define model dictionaries
    clinical_model_configs = {
        'lr': {'model': sk_LogisticRegression, 'params': {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'random_state': 42}, 'scale': True},
        'svm': {'model': SVC, 'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42}, 'scale': True},
        'rf': {'model': RandomForestClassifier, 'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}, 'scale': False},
        'gb': {'model': GradientBoostingClassifier, 'params': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}, 'scale': False}
    }
    
    pathology_model_configs = {
        'lr': {'model': sk_LogisticRegression, 'params': {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'random_state': 42}, 'scale': True},
        'svm': {'model': SVC, 'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42}, 'scale': True},
        'rf': {'model': RandomForestClassifier, 'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}, 'scale': False}
    }
    
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
        
        # Initialize and train each clinical model
        clinical_results = {}
        for name, config in clinical_model_configs.items():
            model = config['model'](**config['params'])
            scale = config['scale']
            
            result = train_eval_sklearn_model(
                model, X_clinical_train, y_clinical_train, 
                X_clinical_test, y_clinical_test, 
                f"Clinical {name.upper()}", scale_features=scale
            )
            
            clinical_results[name] = result
            
            # Store performance metrics for this fold
            clinical_models_performance[name]['accuracy'].append(result['accuracy'])
            clinical_models_performance[name]['bacc'].append(result['balanced_accuracy'])
            if result.get('auc') is not None:
                clinical_models_performance[name]['auc'].append(result['auc'])
        
        # --- 4.4. Train Pathology Models ---
        log_output("\n--- Training Pathology Models ---")
        
        # Initialize and train each pathology model
        pathology_results = {}
        for name, config in pathology_model_configs.items():
            model = config['model'](**config['params'])
            scale = config['scale']
            
            metrics, dump = eval_sklearn_classifier(
                model, torch.tensor(X_pathology_train), torch.tensor(y_pathology_train),
                torch.tensor(X_pathology_test), torch.tensor(y_pathology_test),
                prefix=f"pathology_{name}_", scale_features=scale
            )
            
            # Convert to format compatible with clinical results
            result = {
                'model': model,
                'scaler': dump.get('scaler'),
                'accuracy': metrics.get(f"pathology_{name}_acc", 0),
                'balanced_accuracy': metrics.get(f"pathology_{name}_bacc", 0),
                'auc': metrics.get(f"pathology_{name}_auroc", None),
                'predictions': dump['preds_all'],
                'probabilities': dump['probs_all']
            }
            
            pathology_results[name] = result
            
            log_output(f"Pathology {name.upper()} - Accuracy: {result['accuracy']:.4f}, "
                      f"Balanced Acc: {result['balanced_accuracy']:.4f}")
            
            # Store performance metrics for this fold
            pathology_models_performance[name]['accuracy'].append(result['accuracy'])
            pathology_models_performance[name]['bacc'].append(result['balanced_accuracy'])
            if result.get('auc') is not None:
                pathology_models_performance[name]['auc'].append(result['auc'])
            
        # --- 4.5. Aggregate Pathology Predictions at Slide Level ---
        log_output("\n--- Aggregating Pathology Predictions at Slide Level for Each Model ---")
        
        # Store slide-level predictions for each pathology model
        slide_level_results = {}
        for name in pathology_model_configs.keys():
            slide_ids, y_true_slides, y_pred_slides, y_prob_slides = get_slide_level_predictions(
                test_items,
                pathology_results[name]['predictions'],
                pathology_results[name]['probabilities']
            )
            
            # Calculate slide-level metrics
            slide_accuracy = accuracy_score(y_true_slides, y_pred_slides)
            slide_bacc = balanced_accuracy_score(y_true_slides, y_pred_slides)
            
            slide_level_results[name] = {
                'slide_ids': slide_ids,
                'true_labels': y_true_slides,
                'predictions': y_pred_slides,
                'probabilities': y_prob_slides,
                'accuracy': slide_accuracy,
                'balanced_accuracy': slide_bacc
            }
            
            log_output(f"Slide-level {name.upper()} - Accuracy: {slide_accuracy:.4f}, "
                      f"Balanced Acc: {slide_bacc:.4f}")
            
            # Store slide-level performance for this fold
            slide_level_performance[name]['accuracy'].append(slide_accuracy)
            slide_level_performance[name]['bacc'].append(slide_bacc)
        
        # --- 4.6. Late Fusion: Try All Combinations ---
        log_output("\n--- Late Fusion: Evaluating All Model Combinations ---")
        
        # Generate all combinations of clinical and pathology models
        for clin_name, path_name in itertools.product(clinical_model_configs.keys(), pathology_model_configs.keys()):
            combo_name = f"{clin_name}+{path_name}"
            log_output(f"\n--- Evaluating combination: {combo_name} ---")
            
            # Get slide-level predictions for this pathology model
            slide_ids = slide_level_results[path_name]['slide_ids']
            slide_prob_map = {sid: prob for sid, prob in zip(slide_ids, slide_level_results[path_name]['probabilities'])}
            slide_pred_map = {sid: pred for sid, pred in zip(slide_ids, slide_level_results[path_name]['predictions'])}
            slide_truth_map = {sid: truth for sid, truth in zip(slide_ids, slide_level_results[path_name]['true_labels'])}
            
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
                            # Binary classification - use probability of positive class
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
                    true_label = y_clinical_test[i]  # Use clinical ground truth
                    true_labels.append(true_label)
            
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
                    fusion_performance[combo_name]['auc'].append(fusion_auc)
                except:
                    fusion_auc = None
                    log_output("Could not compute fusion AUC (possibly only one class present)")
                
                log_output(f"Fusion accuracy: {fusion_acc:.4f}")
                log_output(f"Fusion balanced accuracy: {fusion_bacc:.4f}")
                
                # Compare with individual modalities on common patients
                clinical_only_preds = [1 if p >= 0.5 else 0 for p in clinical_probs]
                clinical_only_acc = accuracy_score(true_labels, clinical_only_preds)
                clinical_only_bacc = balanced_accuracy_score(true_labels, clinical_only_preds)
                
                pathology_only_preds = [1 if p >= 0.5 else 0 for p in pathology_probs]
                pathology_only_acc = accuracy_score(true_labels, pathology_only_preds)
                pathology_only_bacc = balanced_accuracy_score(true_labels, pathology_only_preds)
                
                log_output(f"On common patients - Clinical: {clinical_only_acc:.4f}, Pathology: {pathology_only_acc:.4f}, Fusion: {fusion_acc:.4f}")
                
                # Store fusion performance for this combination in this fold
                fusion_performance[combo_name]['accuracy'].append(fusion_acc)
                fusion_performance[combo_name]['bacc'].append(fusion_bacc)
                fusion_performance[combo_name]['clinical_acc'].append(clinical_only_acc)
                fusion_performance[combo_name]['pathology_acc'].append(pathology_only_acc)
                fusion_performance[combo_name]['n_patients'].append(len(common_patients))
            else:
                log_output(f"No common patients found for combination {combo_name} in this fold!")
    
    # ------ 5. Calculate & Report Average Performance Across Folds ------
    
    # --- 5.1. Clinical Models Performance ---
    log_output("\n=== Clinical Models Performance (mean ± std) ===")
    clinical_summary = []
    
    for name in clinical_model_configs.keys():
        acc_mean = np.mean(clinical_models_performance[name]['accuracy'])
        acc_std = np.std(clinical_models_performance[name]['accuracy'])
        bacc_mean = np.mean(clinical_models_performance[name]['bacc'])
        bacc_std = np.std(clinical_models_performance[name]['bacc'])
        
        row = {
            'model': f"Clinical {name.upper()}",
            'accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'balanced_accuracy': f"{bacc_mean:.4f} ± {bacc_std:.4f}",
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'bacc_mean': bacc_mean, 
            'bacc_std': bacc_std
        }
        
        if clinical_models_performance[name].get('auc'):
            auc_mean = np.mean(clinical_models_performance[name]['auc'])
            auc_std = np.std(clinical_models_performance[name]['auc'])
            row['auc'] = f"{auc_mean:.4f} ± {auc_std:.4f}"
            row['auc_mean'] = auc_mean
            row['auc_std'] = auc_std
            
            log_output(f"Clinical {name.upper()} - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}, AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        else:
            log_output(f"Clinical {name.upper()} - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}")
        
        clinical_summary.append(row)
    
    # --- 5.2. Pathology Models Performance (Patch Level) ---
    log_output("\n=== Pathology Models Performance - Patch Level (mean ± std) ===")
    pathology_summary = []
    
    for name in pathology_model_configs.keys():
        acc_mean = np.mean(pathology_models_performance[name]['accuracy'])
        acc_std = np.std(pathology_models_performance[name]['accuracy'])
        bacc_mean = np.mean(pathology_models_performance[name]['bacc'])
        bacc_std = np.std(pathology_models_performance[name]['bacc'])
        
        row = {
            'model': f"Pathology {name.upper()} (Patch)",
            'accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'balanced_accuracy': f"{bacc_mean:.4f} ± {bacc_std:.4f}",
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'bacc_mean': bacc_mean,
            'bacc_std': bacc_std
        }
        
        if pathology_models_performance[name].get('auc'):
            auc_mean = np.mean(pathology_models_performance[name]['auc'])
            auc_std = np.std(pathology_models_performance[name]['auc'])
            row['auc'] = f"{auc_mean:.4f} ± {auc_std:.4f}"
            row['auc_mean'] = auc_mean
            row['auc_std'] = auc_std
            
            log_output(f"Pathology {name.upper()} (Patch) - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}, AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        else:
            log_output(f"Pathology {name.upper()} (Patch) - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}")
        
        pathology_summary.append(row)
    
    # --- 5.3. Pathology Models Performance (Slide Level) ---
    log_output("\n=== Pathology Models Performance - Slide Level (mean ± std) ===")
    slide_level_summary = []
    
    for name in pathology_model_configs.keys():
        acc_mean = np.mean(slide_level_performance[name]['accuracy'])
        acc_std = np.std(slide_level_performance[name]['accuracy'])
        bacc_mean = np.mean(slide_level_performance[name]['bacc'])
        bacc_std = np.std(slide_level_performance[name]['bacc'])
        
        row = {
            'model': f"Pathology {name.upper()} (Slide)",
            'accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'balanced_accuracy': f"{bacc_mean:.4f} ± {bacc_std:.4f}",
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'bacc_mean': bacc_mean,
            'bacc_std': bacc_std
        }
        
        log_output(f"Pathology {name.upper()} (Slide) - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                  f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}")
        
        slide_level_summary.append(row)
    
    # --- 5.4. Fusion Models Performance ---
    log_output("\n=== Fusion Models Performance (mean ± std) ===")
    fusion_summary = []
    
    for combo_name in fusion_performance.keys():
        if not fusion_performance[combo_name]['accuracy']:
            # Skip combinations that didn't have results (no common patients)
            continue
            
        acc_mean = np.mean(fusion_performance[combo_name]['accuracy'])
        acc_std = np.std(fusion_performance[combo_name]['accuracy'])
        bacc_mean = np.mean(fusion_performance[combo_name]['bacc'])
        bacc_std = np.std(fusion_performance[combo_name]['bacc'])
        
        clinical_acc_mean = np.mean(fusion_performance[combo_name]['clinical_acc'])
        clinical_acc_std = np.std(fusion_performance[combo_name]['clinical_acc'])
        
        pathology_acc_mean = np.mean(fusion_performance[combo_name]['pathology_acc'])
        pathology_acc_std = np.std(fusion_performance[combo_name]['pathology_acc'])
        
        avg_patients = np.mean(fusion_performance[combo_name]['n_patients'])
        
        row = {
            'model': f"Fusion {combo_name}",
            'accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'balanced_accuracy': f"{bacc_mean:.4f} ± {bacc_std:.4f}",
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'bacc_mean': bacc_mean,
            'bacc_std': bacc_std,
            'clinical_acc': f"{clinical_acc_mean:.4f} ± {clinical_acc_std:.4f}",
            'pathology_acc': f"{pathology_acc_mean:.4f} ± {pathology_acc_std:.4f}",
            'clinical_acc_mean': clinical_acc_mean,
            'pathology_acc_mean': pathology_acc_mean,
            'avg_patients': avg_patients
        }
        
        if fusion_performance[combo_name].get('auc'):
            auc_mean = np.mean(fusion_performance[combo_name]['auc'])
            auc_std = np.std(fusion_performance[combo_name]['auc'])
            row['auc'] = f"{auc_mean:.4f} ± {auc_std:.4f}"
            row['auc_mean'] = auc_mean
            row['auc_std'] = auc_std
            
            log_output(f"Fusion {combo_name} - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}, AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        else:
            log_output(f"Fusion {combo_name} - Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, "
                      f"Balanced Acc: {bacc_mean:.4f} ± {bacc_std:.4f}")
        
        log_output(f"  Clinical Only: {clinical_acc_mean:.4f} ± {clinical_acc_std:.4f}, "
                  f"Pathology Only: {pathology_acc_mean:.4f} ± {pathology_acc_std:.4f}, "
                  f"Avg Patients: {avg_patients:.1f}")
        
        fusion_summary.append(row)
    
    # --- 5.5. Find Best Combinations ---
    if fusion_summary:
        log_output("\n=== Best Performing Combinations ===")
        
        # Create DataFrame for easier sorting and reporting
        fusion_df = pd.DataFrame(fusion_summary)
        
        # Sort by accuracy (descending)
        best_by_acc = fusion_df.sort_values('accuracy_mean', ascending=False).head(3)
        log_output("\nTop 3 combinations by accuracy:")
        for _, row in best_by_acc.iterrows():
            log_output(f"{row['model']}: {row['accuracy']} (vs Clinical: {row['clinical_acc']}, Pathology: {row['pathology_acc']})")
        
        # Sort by balanced accuracy (descending)
        best_by_bacc = fusion_df.sort_values('bacc_mean', ascending=False).head(3)
        log_output("\nTop 3 combinations by balanced accuracy:")
        for _, row in best_by_bacc.iterrows():
            log_output(f"{row['model']}: {row['balanced_accuracy']} (vs Clinical: {row['clinical_acc']}, Pathology: {row['pathology_acc']})")
        
        # Check if any fusion model beat both its constituent clinical and pathology models
        log_output("\nFusion models that outperformed both constituent models:")
        for _, row in fusion_df.iterrows():
            if row['accuracy_mean'] > row['clinical_acc_mean'] and row['accuracy_mean'] > row['pathology_acc_mean']:
                log_output(f"{row['model']}: {row['accuracy']} (vs Clinical: {row['clinical_acc']}, Pathology: {row['pathology_acc']})")
    
    # --- 5.6. Save Results to CSV ---
    log_output("\n=== Saving Results to CSV ===")
    
    # Combine all results into one dataframe
    all_results = pd.concat([
        pd.DataFrame(clinical_summary),
        pd.DataFrame(pathology_summary),
        pd.DataFrame(slide_level_summary),
        pd.DataFrame(fusion_summary) if fusion_summary else pd.DataFrame()
    ])
    
    # Save to CSV
    results_file = "all_combinations_performance.csv"
    all_results.to_csv(results_file, index=False)
    log_output(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()