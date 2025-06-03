import torch
import numpy as np
import pickle
import os
import pandas as pd # For eval_fewshot results
import logging # For cleaner logging from original functions
from tqdm import tqdm
from typing import Optional, Dict, Any, Union, List, Tuple
import itertools # For manual grid search

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as sk_KNeighborsClassifier
from sklearn.model_selection import GridSearchCV # Added for hyperparameter tuning
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

# Silence repeated convergence warnings from scikit-learn logistic regression.
simplefilter("ignore", category=ConvergenceWarning)

# --- 0. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}") # Moved to log_output in main

# Seed for reproducibility (optional, but good practice)
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
# For sklearn classifiers that have random_state
sklearn_random_state = 42


log_file_path = "model_evaluation_combined_log.txt"
# Clear log file at the start of a new run (optional)
with open(log_file_path, "w") as f:
    f.write(f"Log started at {pd.Timestamp.now(tz='America/Toronto')}\n")


def log_output(message):
    """Simple logger to print and write to a file."""
    print(message)
    with open(log_file_path, "a") as f:
        f.write(str(message) + "\n")

# Configure basic logging for functions that use 'logging' module
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Metric Calculation Function (get_eval_metrics - updated slightly for clarity) ---
def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {}, # Allows passing custom args to roc_auc_score if needed
) -> Dict[str, Any]:
    targets_all = np.asarray(targets_all)
    preds_all = np.asarray(preds_all)

    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)

    # Determine labels present in the data for classification report
    present_labels = np.unique(np.concatenate((targets_all, preds_all))).astype(int)
    present_labels.sort() # Ensure consistent order
    target_names = [f"class {i}" for i in present_labels]

    try:
        cls_rep = classification_report(
            targets_all,
            preds_all,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        log_output(f"Warning: Could not generate classification report fully for prefix '{prefix}': {e}")
        cls_rep = {"weighted avg": {"f1-score": 0.0}}
        # Ensure dummy entries for robustness if downstream code expects them
        for label_idx in present_labels:
            class_name = f"class {label_idx}"
            if class_name not in cls_rep:
                cls_rep[class_name] = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
        if "macro avg" not in cls_rep:
            cls_rep["macro avg"] = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
        if "weighted avg" not in cls_rep:
            cls_rep["weighted avg"] = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}


    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep.get("weighted avg", {}).get("f1-score", 0.0),
    }

    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    if probs_all is not None:
        probs_all = np.asarray(probs_all)
        # Ensure y_true for roc_auc_score has at least two classes
        if len(np.unique(targets_all)) > 1:
            try:
                if probs_all.ndim == 1: # Binary case, probs of positive class
                    roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
                elif probs_all.shape[1] == 2: # Binary, (N, 2) shape, use probs of positive class
                    roc_auc = roc_auc_score(targets_all, probs_all[:, 1], **roc_kwargs)
                elif probs_all.shape[1] > 2: # Multiclass case
                    # Ensure labels are consistent if using roc_kwargs like average=None
                    roc_auc = roc_auc_score(targets_all, probs_all, multi_class='ovo', average='macro', labels=np.unique(targets_all), **roc_kwargs)
                else: # Ambiguous case or single class in probs
                    log_output(f"Warning: AUROC not computed for prefix '{prefix}' due to ambiguous probability shape or insufficient classes in probs_all.")
                    roc_auc = np.nan
                eval_metrics[f"{prefix}auroc"] = roc_auc
            except ValueError as ve:
                log_output(f"Warning: Could not compute AUROC for prefix '{prefix}': {ve}. Assigning NaN.")
                eval_metrics[f"{prefix}auroc"] = np.nan
        else:
            log_output(f"Warning: AUROC not computed for prefix '{prefix}' because only one class present in y_true.")
            eval_metrics[f"{prefix}auroc"] = np.nan
    return eval_metrics

# --- New Function for Hyperparameter Tuning and Evaluation ---
def tune_and_evaluate_classifier(
    base_estimator,
    param_grid: Dict[str, Any],
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    scoring: str = 'balanced_accuracy',
    cv: int = 3, # Reduced CV for faster example, recommend 5 for robust results
    prefix: str = "",
    scale_features: bool = False
):
    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_feats.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    if scale_features:
        log_output(f"Scaling features for {prefix}tuning and evaluation...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    log_output(f"--- Starting GridSearchCV for {prefix}{type(base_estimator).__name__} ---")
    log_output(f"Parameter grid: {param_grid}")
    log_output(f"Scoring metric for CV: {scoring}, CV folds: {cv}")

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1, # Use all available cores
        verbose=1 # Logs verbosity from GridSearchCV
    )
    grid_search.fit(X_train, y_train)

    log_output(f"Best parameters found for {prefix}: {grid_search.best_params_}")
    log_output(f"Best CV {scoring} score for {prefix}: {grid_search.best_score_:.4f}")

    best_classifier = grid_search.best_estimator_

    log_output(f"Predicting with tuned {prefix}classifier on test set...")
    preds_all = best_classifier.predict(X_test)
    probs_all = None
    if hasattr(best_classifier, "predict_proba"):
        try:
            probs_all = best_classifier.predict_proba(X_test)
        except Exception as e:
            log_output(f"Could not get probabilities for {prefix}: {e}")
    else:
        log_output(f"Tuned classifier for {prefix} does not have predict_proba method.")

    # roc_kwargs can be passed if specific settings are needed beyond get_eval_metrics defaults
    # For binary classification, get_eval_metrics handles common cases automatically.
    metrics_roc_kwargs = {}

    metrics = get_eval_metrics(
        y_test, preds_all, probs_all,
        get_report=True, prefix=prefix, roc_kwargs=metrics_roc_kwargs
    )
    dump = {
        "preds_all": preds_all.tolist(), "probs_all": probs_all.tolist() if probs_all is not None else None,
        "targets_all": y_test.tolist(),
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "classifier_details": str(best_classifier)
    }
    return metrics, dump

# Placeholder for original eval_sklearn_classifier if needed elsewhere,
# but for tuning, we'll use the new function.
def eval_sklearn_classifier(
    classifier, train_feats: torch.Tensor, train_labels: torch.Tensor,
    test_feats: torch.Tensor, test_labels: torch.Tensor,
    prefix: str = "", scale_features: bool = False
):
    # This function is kept for compatibility if used by other parts of your code
    # For new tuning evaluations, tune_and_evaluate_classifier is preferred.
    log_output(f"Executing original eval_sklearn_classifier for {prefix}{type(classifier).__name__}. Consider using tune_and_evaluate_classifier for hyperparameter search.")
    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_feats.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    if scale_features:
        log_output(f"Scaling features for {prefix} evaluation...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    log_output(f"Training {prefix} classifier ({type(classifier).__name__})...")
    classifier.fit(X_train, y_train)

    log_output(f"Predicting with {prefix} classifier...")
    preds_all = classifier.predict(X_test)
    probs_all = None
    if hasattr(classifier, "predict_proba"):
        probs_all = classifier.predict_proba(X_test)
    else:
        log_output(f"Classifier for {prefix} does not have predict_proba method.")
    
    metrics_roc_kwargs = {} # Default, get_eval_metrics handles common cases

    metrics = get_eval_metrics(y_test, preds_all, probs_all, get_report=True, prefix=prefix, roc_kwargs=metrics_roc_kwargs)
    dump = {
        "preds_all": preds_all.tolist(), "probs_all": probs_all.tolist() if probs_all is not None else None,
        "targets_all": y_test.tolist(),
        "classifier_details": str(classifier)
    }
    return metrics, dump


# --- Main Execution ---
if __name__ == '__main__':
    log_output(f"\n--- Evaluation Run Started: {pd.Timestamp.now(tz='America/Toronto')} ---")
    log_output(f"Using device: {device}")
    log_output(f"Seed value: {seed_value}")

    # --- Load UNI Model (Optional, for context if needed elsewhere) ---
    # (Assuming uni, eval_linear_probe, eval_knn, ProtoNet, eval_fewshot are defined elsewhere if not provided)
    try:
        # from uni import get_encoder # If you need this
        pass
    except ImportError:
        log_output("Could not import 'uni'. This is fine if using pre-extracted features.")
    except Exception as e:
        log_output(f"Error loading UNI model: {e}")

    # --- Load Pre-extracted Features ---
    train_pkl_path = 'train_features_portal_tract_separated.pkl' # Replace with your actual path
    test_pkl_path = 'test_features_portal_tract_separated.pkl'   # Replace with your actual path

    # Dummy data generation if PKL files are not found (for testing the script structure)
    if not (os.path.exists(train_pkl_path) and os.path.exists(test_pkl_path)):
        log_output(f"WARNING: Feature PKL file(s) not found. USING DUMMY DATA for demonstration.")
        log_output(f"Checked train: {train_pkl_path}")
        log_output(f"Checked test: {test_pkl_path}")
        
        # Create dummy data
        num_train_samples = 200
        num_test_samples = 50
        num_features = 128 # Example feature dimension
        
        dummy_train_embeddings = np.random.rand(num_train_samples, num_features).astype(np.float32)
        dummy_train_labels = np.random.randint(0, 2, num_train_samples).astype(np.int64) # Binary labels
        dummy_test_embeddings = np.random.rand(num_test_samples, num_features).astype(np.float32)
        dummy_test_labels = np.random.randint(0, 2, num_test_samples).astype(np.int64)

        loaded_train_features = {'embeddings': dummy_train_embeddings, 'labels': dummy_train_labels}
        loaded_test_features = {'embeddings': dummy_test_embeddings, 'labels': dummy_test_labels}
        
        # Save dummy data as PKL for subsequent runs if desired (optional)
        # with open(train_pkl_path, 'wb') as f: pickle.dump(loaded_train_features, f)
        # with open(test_pkl_path, 'wb') as f: pickle.dump(loaded_test_features, f)
    else:
        log_output("Loading features from PKL files.")
        with open(train_pkl_path, 'rb') as f:
            loaded_train_features = pickle.load(f)
        with open(test_pkl_path, 'rb') as f:
            loaded_test_features = pickle.load(f)

    train_feats = torch.Tensor(loaded_train_features['embeddings']).to(device)
    train_labels = torch.Tensor(loaded_train_features['labels']).type(torch.long).to(device)
    test_feats = torch.Tensor(loaded_test_features['embeddings']).to(device)
    test_labels = torch.Tensor(loaded_test_features['labels']).type(torch.long).to(device)

    log_output(f"Data Loaded. Train features: {train_feats.shape}, Test features: {test_feats.shape}")
    log_output(f"Unique train labels: {torch.unique(train_labels).cpu().numpy()}")
    log_output(f"Unique test labels: {torch.unique(test_labels).cpu().numpy()}")
    log_output(f"Train label distribution: {np.bincount(train_labels.cpu().numpy())}")
    log_output(f"Test label distribution: {np.bincount(test_labels.cpu().numpy())}")


    # Define scoring metric for GridSearchCV
    # Common choices: 'accuracy', 'balanced_accuracy', 'f1_weighted', 'roc_auc'
    # For potentially imbalanced binary classification, 'balanced_accuracy' or 'f1_weighted' or 'roc_auc' are good.
    cv_scoring_metric = 'balanced_accuracy'
    num_cv_folds = 3 # Use 5 for more robust results, 3 for quicker runs

    # --- 1. Linear Probing (Logistic Regression) Evaluation with Hyperparameter Tuning ---
    log_output("\n\n--- Starting Tuned Linear Probing (Logistic Regression) Evaluation ---")
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'], # saga supports l1, liblinear supports l1/l2
        'class_weight': ['balanced', None],
        'max_iter': [100000, 500000, 1000000, 2000000, 10000000] # Increased max_iter
    }
    # Filter grid for solver-penalty compatibility
    lr_param_grid_filtered = []
    for p in lr_param_grid['penalty']:
        for s in lr_param_grid['solver']:
            if (p == 'l1' and s in ['liblinear', 'saga']) or \
               (p == 'l2' and s in ['liblinear', 'saga']): # saga also supports l2
                for c_val in lr_param_grid['C']:
                    for cw in lr_param_grid['class_weight']:
                        for mi in lr_param_grid['max_iter']:
                             lr_param_grid_filtered.append({
                                 'C': [c_val], 'penalty': [p], 'solver': [s],
                                 'class_weight': [cw], 'max_iter': [mi],
                                 'random_state': [sklearn_random_state]
                             })

    # Note: The original `eval_linear_probe` call is replaced.
    # If `eval_linear_probe` had specific logic beyond a standard Logistic Regression,
    # that logic would need to be integrated or handled separately.
    if lr_param_grid_filtered: # Check if grid is not empty after filtering
        linprobe_metrics, linprobe_dump = tune_and_evaluate_classifier(
            base_estimator=sk_LogisticRegression(random_state=sklearn_random_state),
            param_grid=lr_param_grid_filtered,
            train_feats=train_feats, train_labels=train_labels,
            test_feats=test_feats, test_labels=test_labels,
            scoring=cv_scoring_metric, cv=num_cv_folds,
            prefix="tuned_linprobe_",
            scale_features=True # Scaling is generally good for Logistic Regression
        )
        log_output("Tuned Linear Probing (Logistic Regression) Metrics:")
        for k, v in linprobe_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")
    else:
        log_output("Skipping Logistic Regression tuning due to empty parameter grid after filtering.")

    # --- 2. k-NN Evaluation with Hyperparameter Tuning ---
    log_output("\n\n--- Starting Tuned k-NN Evaluation ---")
    knn_param_grid = {
        'n_neighbors': [3, 7, 11, 15, 19, 23, 30],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2] # Only used when metric is 'minkowski' (p=1 is manhattan, p=2 is euclidean)
    }
    # Filter grid for metric-p compatibility
    knn_param_grid_filtered = []
    for n in knn_param_grid['n_neighbors']:
        for w in knn_param_grid['weights']:
            for m in knn_param_grid['metric']:
                if m == 'minkowski':
                    for p_val in knn_param_grid['p']:
                        knn_param_grid_filtered.append({
                            'n_neighbors': [n], 'weights': [w], 'metric': [m], 'p': [p_val]
                        })
                else: # p is ignored for other metrics
                     knn_param_grid_filtered.append({
                        'n_neighbors': [n], 'weights': [w], 'metric': [m]
                    })

    # Note: The original `eval_knn` call is replaced for the k-NN part.
    # The "Basic Prototype (from k-NN func)" part of the original eval_knn is not covered here.
    # If that's needed, it would require separate implementation or modification of eval_knn.
    if knn_param_grid_filtered:
        knn_metrics, knn_dump = tune_and_evaluate_classifier(
            base_estimator=sk_KNeighborsClassifier(n_jobs=-1), # n_jobs for KNeighborsClassifier constructor
            param_grid=knn_param_grid_filtered,
            train_feats=train_feats, train_labels=train_labels,
            test_feats=test_feats, test_labels=test_labels,
            scoring=cv_scoring_metric, cv=num_cv_folds,
            prefix="tuned_knn_",
            scale_features=True # Scaling is often beneficial for k-NN
        )
        log_output(f"Tuned k-NN Metrics:")
        for k, v in knn_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")
    else:
        log_output("Skipping k-NN tuning due to empty parameter grid after filtering.")


    # --- 3. Few-Shot Learning Evaluation (Example of manual tuning structure) ---
    # The actual eval_fewshot function is not provided, so this is a structural example.
    # You would need to have `eval_fewshot` defined.
    log_output("\n\n--- Starting Few-Shot (4-shot) Evaluation (Manual Tuning Example) ---")
    # Example: Parameters to tune for eval_fewshot
    # fs_param_grid_manual = {
    #     'n_shot': [1, 4, 8],
    #     'center_feats': [True, False],
    #     'normalize_feats': [True, False],
    #     'average_feats': [True, False] # For prototype calculation within few-shot
    # }
    # best_fs_score = -1.0
    # best_fs_params = {}
    # best_fs_summary_metrics = {}

    # if 'eval_fewshot' in globals(): # Check if the function is defined
    #     keys_fs, values_fs = zip(*fs_param_grid_manual.items())
    #     for v_fs_tuple in itertools.product(*values_fs):
    #         current_fs_params = dict(zip(keys_fs, v_fs_tuple))
    #         log_output(f"Testing Few-Shot with params: {current_fs_params}")
    #         try:
    #             current_fs_results_df, current_fs_summary_metrics = eval_fewshot(
    #                 train_feats=train_feats, train_labels=train_labels,
    #                 test_feats=test_feats, test_labels=test_labels,
    #                 n_iter=50, # Keep other params like n_iter, n_way fixed or tune them too
    #                 n_way=2,
    #                 n_query_eval_full_test=True,
    #                 n_query_episode=15, # Ignored if n_query_eval_full_test is True
    #                 **current_fs_params # Pass tunable params
    #             )
    #             # Choose a metric from current_fs_summary_metrics to optimize, e.g., 'mean_bacc'
    #             score_to_optimize = current_fs_summary_metrics.get('mean_bacc', 0.0)
    #             if score_to_optimize > best_fs_score:
    #                 best_fs_score = score_to_optimize
    #                 best_fs_params = current_fs_params
    #                 best_fs_summary_metrics = current_fs_summary_metrics
    #             log_output(f"Few-Shot with {current_fs_params} -> Score: {score_to_optimize:.4f}")
    #         except Exception as e:
    #             log_output(f"Error during few-shot evaluation with params {current_fs_params}: {e}")
        
    #     if best_fs_params:
    #         log_output(f"Best Few-Shot Parameters: {best_fs_params}")
    #         log_output("Best Few-Shot Aggregated Metrics:")
    #         for k, v in best_fs_summary_metrics.items(): log_output(f"  {k}: {v:.4f}")
    #     else:
    #         log_output("Few-shot tuning did not yield results (or eval_fewshot is not defined).")
    # else:
    #     log_output("eval_fewshot function not defined. Skipping few-shot tuning.")
    # For now, running the original call if eval_fewshot is available
    if 'eval_fewshot' in globals() and callable(globals()['eval_fewshot']):
        log_output("\n\n--- Running Original Few-Shot (4-shot) Evaluation ---")
        fs_results_df, fs_summary_metrics = eval_fewshot(
             train_feats=train_feats, train_labels=train_labels,
             test_feats=test_feats, test_labels=test_labels,
             n_iter=50, n_way=2, n_shot=4,
             n_query_eval_full_test=True, n_query_episode=15,
             center_feats=True, normalize_feats=True, average_feats=True
        )
        log_output("Few-Shot (4-shot) Aggregated Metrics (Original Call):")
        for k, v in fs_summary_metrics.items(): log_output(f"  {k}: {v:.4f}")
    else:
        log_output("Skipping Few-Shot evaluation as 'eval_fewshot' is not defined in the current scope.")


    # --- 4. Standalone Prototypical Network Evaluation (Example of manual tuning) ---
    # The actual ProtoNet class is not provided, so this is a structural example.
    # log_output("\n\n--- Starting Standalone Prototypical Network Evaluation with Manual Tuning ---")
    # protonet_param_grid_manual = {
    #     'metric': ['L2'], # Add 'cosine' if implemented in your ProtoNet
    #     'center_feats': [True, False],
    #     'normalize_feats': [True, False]
    # }
    # best_protonet_score = -1.0  # e.g., for bacc
    # best_protonet_params = {}
    # best_standalone_protonet_metrics = {}

    # if 'ProtoNet' in globals(): # Check if the class is defined
    #     keys_pn, values_pn = zip(*protonet_param_grid_manual.items())
    #     for v_pn_tuple in itertools.product(*values_pn):
    #         current_pn_params = dict(zip(keys_pn, v_pn_tuple))
    #         log_output(f"Testing ProtoNet with params: {current_pn_params}")
    #         try:
    #             proto_clf = ProtoNet(**current_pn_params)
    #             proto_clf.fit(train_feats, train_labels)
    #             test_preds_protonet = proto_clf.predict(test_feats)
    #             current_metrics_protonet = get_eval_metrics(
    #                 test_labels.cpu().numpy(),
    #                 test_preds_protonet.cpu().numpy(),
    #                 probs_all=None, # Assuming simple ProtoNet doesn't give probs
    #                 get_report=True,
    #                 prefix=f"protonet_{'_'.join(f'{k[0]}{str(v)[0]}' for k,v in current_pn_params.items())}_"
    #             )
    #             # Choose a metric to optimize, e.g., bacc
    #             score_to_optimize_pn = current_metrics_protonet.get(next(k for k in current_metrics_protonet if k.endswith('bacc')), 0.0)
    #             if score_to_optimize_pn > best_protonet_score:
    #                 best_protonet_score = score_to_optimize_pn
    #                 best_protonet_params = current_pn_params
    #                 best_standalone_protonet_metrics = current_metrics_protonet
    #             log_output(f"ProtoNet with {current_pn_params} -> Score: {score_to_optimize_pn:.4f}")
    #         except Exception as e:
    #             log_output(f"Error during ProtoNet evaluation with params {current_pn_params}: {e}")

    #     if best_protonet_params:
    #         log_output(f"Best ProtoNet Parameters: {best_protonet_params}")
    #         log_output("Best Standalone Prototypical Network Metrics:")
    #         for k, v in best_standalone_protonet_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")
    #     else:
    #         log_output("ProtoNet tuning did not yield results (or ProtoNet class is not defined).")
    # else:
    #     log_output("ProtoNet class not defined. Skipping ProtoNet tuning.")
    # For now, running the original call if ProtoNet is available
    if 'ProtoNet' in globals() and callable(globals()['ProtoNet']):
        log_output("\n\n--- Running Original Standalone Prototypical Network Evaluation ---")
        proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
        proto_clf.fit(train_feats, train_labels)
        if proto_clf.prototype_embeddings is not None:
            log_output(f"ProtoNet - Prototypes shape: {proto_clf.prototype_embeddings.shape}")
        test_preds_protonet = proto_clf.predict(test_feats)
        standalone_protonet_metrics = get_eval_metrics(
            test_labels.cpu().numpy(), test_preds_protonet.cpu().numpy(),
            probs_all=None, get_report=True, prefix="standalone_protonet_"
        )
        log_output("Standalone Prototypical Network Metrics (Original Call):")
        for k, v in standalone_protonet_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")
    else:
        log_output("Skipping Prototypical Network evaluation as 'ProtoNet' is not defined.")


    # --- 5. SVM Evaluation with Hyperparameter Tuning ---
    log_output("\n\n--- Starting Tuned SVM Evaluation ---")
    svm_param_grid = [
        {'C': [0.001, 0.01, 0.1], 'kernel': ['linear'], 'class_weight': ['balanced', None], 'random_state': [sklearn_random_state], 'probability': [True]},
        {'C': [0.001, 0.01, 0.1], 'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.01, 0.1], 'class_weight': ['balanced', None], 'random_state': [sklearn_random_state], 'probability': [True]},
        # Poly can be slow, added a smaller set
        {'C': [0.001, 0.01, 0.1], 'kernel': ['poly'], 'degree': [2, 3, 5], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced', None], 'random_state': [sklearn_random_state], 'probability': [True]}
    ]

    svm_metrics, svm_dump_data = tune_and_evaluate_classifier(
        base_estimator=SVC(probability=True, random_state=sklearn_random_state), # probability=True for AUROC
        param_grid=svm_param_grid,
        train_feats=train_feats, train_labels=train_labels,
        test_feats=test_feats, test_labels=test_labels,
        scoring=cv_scoring_metric, cv=num_cv_folds,
        prefix="tuned_svm_",
        scale_features=True # Scaling is crucial for SVM
    )
    log_output("Tuned SVM Metrics:")
    for k, v in svm_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")

    # --- 6. Random Forest Evaluation with Hyperparameter Tuning ---
    log_output("\n\n--- Starting Tuned Random Forest Evaluation ---")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False], # Added bootstrap
        'random_state': [sklearn_random_state],
        'n_jobs': [-1] # For RF constructor
    }
    rf_metrics, rf_dump_data = tune_and_evaluate_classifier(
        base_estimator=RandomForestClassifier(random_state=sklearn_random_state, n_jobs=-1),
        param_grid=rf_param_grid,
        train_feats=train_feats, train_labels=train_labels,
        test_feats=test_feats, test_labels=test_labels,
        scoring=cv_scoring_metric, cv=num_cv_folds,
        prefix="tuned_rf_",
        scale_features=False # RF is less sensitive to feature scaling
    )
    log_output("Tuned Random Forest Metrics:")
    for k, v in rf_metrics.items(): log_output(f"  {k}: {v}" if "report" in k else f"  {k}: {v:.4f}")

    log_output(f"\n--- Evaluation Run Ended: {pd.Timestamp.now(tz='America/Toronto')} ---")