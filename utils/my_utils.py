import torch
import numpy as np
import tqdm
import os
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, accuracy_score,
    classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# Setup logging
log_file = "model_evaluation_mix_portal_tract.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

@torch.no_grad()
def extract_features(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary with feature embeddings, labels, image paths, patient IDs, and text labels

    """
    all_embeddings, all_labels = [], []
    all_paths, all_patient_ids, all_text_labels = [], [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters()).device
    
    # Get the dataset to access the file paths
    dataset = dataloader.dataset

    for batch_idx, (batch, target) in tqdm.tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        # Get paths for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + len(batch), len(dataset))
        batch_paths = [dataset.data[i][0] for i in range(start_idx, end_idx)]
        
        # Extract patient IDs and text labels from filenames
        batch_patient_ids = []
        batch_text_labels = []
        for path in batch_paths:
            filename = os.path.basename(path)
            # Parse "response-5585_0001-3987-19049.jpg" format
            parts = filename.split('-')
            if len(parts) >= 2:
                text_label = parts[0]  # "response" or "no_response"
                patient_id = parts[1]  # "5585_0001"
            else:
                text_label = "unknown"
                patient_id = "unknown"
            
            batch_patient_ids.append(patient_id)
            batch_text_labels.append(text_label)
        
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_paths.extend(batch_paths)
        all_patient_ids.extend(batch_patient_ids)
        all_text_labels.extend(batch_text_labels)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "paths": all_paths,
        "patient_ids": all_patient_ids,
        "text_labels": all_text_labels
    }

    return asset_dict

def get_eval_metrics(targets_all, preds_all, probs_all=None, get_report=True, prefix=""):
    """Calculate evaluation metrics"""
    targets_all = np.asarray(targets_all)
    preds_all = np.asarray(preds_all)
    
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    
    present_labels = np.unique(np.concatenate((targets_all, preds_all))).astype(int)
    target_names = [f"class {i}" for i in present_labels]
    
    cls_rep = classification_report(
        targets_all, preds_all, labels=present_labels, target_names=target_names,
        output_dict=True, zero_division=0
    )
    
    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }
    
    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep
    
    if probs_all is not None:
        probs_all = np.asarray(probs_all)
        if len(np.unique(targets_all)) > 1:
            try:
                if probs_all.ndim == 1:
                    roc_auc = roc_auc_score(targets_all, probs_all)
                elif probs_all.shape[1] == 2:
                    roc_auc = roc_auc_score(targets_all, probs_all[:, 1])
                else:
                    roc_auc = roc_auc_score(targets_all, probs_all, multi_class='ovo', average='macro')
                eval_metrics[f"{prefix}auroc"] = roc_auc
            except ValueError as ve:
                log_output(f"Warning: Could not compute AUROC for {prefix}: {ve}")
                eval_metrics[f"{prefix}auroc"] = np.nan
    
    return eval_metrics

def eval_sklearn_classifier(classifier, train_feats, train_labels, test_feats, test_labels, prefix="", scale_features=False):
    """Evaluate sklearn classifier with optimal hyperparameters"""
    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_feats.cpu().numpy()
    y_test = test_labels.cpu().numpy()
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Make predictions
    preds_all = classifier.predict(X_test)
    probs_all = None
    if hasattr(classifier, "predict_proba"):
        try:
            probs_all = classifier.predict_proba(X_test)
        except Exception as e:
            log_output(f"Could not get probabilities for {prefix}: {e}")
    
    # Calculate metrics
    metrics = get_eval_metrics(y_test, preds_all, probs_all, get_report=True, prefix=prefix)
    
    dump = {
        "preds_all": preds_all,
        "probs_all": probs_all,
        "targets_all": y_test,
        "classifier": classifier,
        "scaler": scaler
    }
    
    return metrics, dump