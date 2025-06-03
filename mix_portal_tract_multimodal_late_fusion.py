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

# ──────────────────────────── 1. Reproducibility ─────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
from uni import get_encoder
model, transform = get_encoder(enc_name="uni2-h", device=device)

# ──────────────────────────── 6. DataLoaders ────────────────────────────────
BATCH = 16
train_dataset = PatchDataset(train_data, transform)
test_dataset  = PatchDataset(test_data, transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                      num_workers=4)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False,
                      num_workers=4)

# ──────────────────────────── 7. Image index CSV (optional) ──────────────────
def make_df(items):
    paths, labels = zip(*items)
    classes = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({"image_path": paths, "label": labels, "class": classes})

csv_out = Path("index_path_mix_portal_tract.csv")
pd.concat([make_df(train_data), make_df(test_data)]).to_csv(csv_out, index_label="idx")
print(f"[INFO] saved slide index → {csv_out}")


# Extract features
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
train_features = extract_patch_features_from_dataloader(model, train_dataloader)

# Save train_features and test_features to files
with open('train_features_mix_portal_tract.pkl', 'wb') as f:
    pickle.dump(train_features, f)

test_features = extract_patch_features_from_dataloader(model, test_dataloader)
with open('test_features_mix_portal_tract.pkl', 'wb') as f:
    pickle.dump(test_features, f)

# Load train_features and test_features from files
with open('train_features_mix_portal_tract.pkl', 'rb') as f:
    loaded_train_features = pickle.load(f)

with open('test_features_mix_portal_tract.pkl', 'rb') as f:
    loaded_test_features = pickle.load(f)

# Convert to tensors
train_feats = torch.Tensor(loaded_train_features['embeddings'])
train_labels = torch.Tensor(loaded_train_features['labels']).type(torch.long)
test_feats = torch.Tensor(loaded_test_features['embeddings'])
test_labels = torch.Tensor(loaded_test_features['labels']).type(torch.long)

# Setup logging
log_file = "model_evaluation_mix_portal_tract.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

log_output(f"\n\n\nTime now is: {pd.Timestamp.now()}\n\n\n")

# ──────────────────────────── 8. Evaluation Functions ────────────────────────
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

# Evaluation methods from original UNI
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics as uni_get_eval_metrics, print_metrics

# ──────────────────────────── 9. Original Model Evaluations ──────────────────
# Linear probe evaluation
log_output("\n--- Linear Probe Evaluation ---")
linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
    train_feats=train_feats, train_labels=train_labels,
    valid_feats=None, valid_labels=None,
    test_feats=test_feats, test_labels=test_labels,
    max_iter=1000, verbose=True,
)
log_output("Linear Probe Evaluation Metrics:")
log_output(str(linprobe_eval_metrics))

# KNN evaluation
log_output("\n--- KNN Evaluation ---")
knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    center_feats=True, normalize_feats=True, n_neighbors=7
)
log_output("KNN Evaluation Metrics:")
log_output(str(knn_eval_metrics))
log_output("ProtoNet Evaluation Metrics:")
log_output(str(proto_eval_metrics))

# Few-shot evaluation
log_output("\n--- Few-Shot Evaluation ---")
fewshot_episodes, fewshot_dump = eval_fewshot(
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    n_iter=500, n_way=2, n_shot=4, n_query=test_feats.shape[0],
    center_feats=True, normalize_feats=True, average_feats=True,
)
log_output("Few Shot Episodes:")
log_output(str(fewshot_episodes))

# ProtoNet evaluation
log_output("\n--- ProtoNet Evaluation ---")
proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
proto_clf.fit(train_feats, train_labels, verbose=True)
log_output('\nWhat our prototypes look like')
log_output(str(proto_clf.prototype_embeddings.shape))

test_pred = proto_clf.predict(test_feats)
eval_metrics = uni_get_eval_metrics(test_labels, test_pred, get_report=False)
log_output("ProtoNet Evaluation Metrics:")
log_output(str(eval_metrics))

# ──────────────────────────── 10. Additional Model Evaluations ───────────────

# Support Vector Machine with optimal hyperparameters for this dataset
log_output("\n--- SVM Evaluation (Optimal Parameters) ---")
# Best params for high-dimensional data (1536 features) with ~4K samples
svm_classifier = SVC(
    C=1.0,                    # Good balance for regularization
    kernel='rbf',             # RBF works well for high-dimensional data
    gamma='scale',            # Automatic scaling
    class_weight='balanced',  # Handle class imbalance
    probability=True,         # Enable probability predictions
    random_state=42
)

svm_metrics, svm_dump = eval_sklearn_classifier(
    classifier=svm_classifier,
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    prefix="svm_", scale_features=True
)
log_output("SVM Metrics:")
for k, v in svm_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")

# Random Forest with optimal hyperparameters
log_output("\n--- Random Forest Evaluation (Optimal Parameters) ---")
# Optimal params for ~4K samples with 1536 features
rf_classifier = RandomForestClassifier(
    n_estimators=200,         # Good balance between performance and speed
    max_depth=10,             # Prevent overfitting with high-dim data
    min_samples_split=10,     # Conservative splitting for stability
    min_samples_leaf=4,       # Prevent overfitting
    max_features='sqrt',      # Good for high-dimensional data
    class_weight='balanced',  # Handle class imbalance
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_metrics, rf_dump = eval_sklearn_classifier(
    classifier=rf_classifier,
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    prefix="rf_", scale_features=False
)
log_output("Random Forest Metrics:")
for k, v in rf_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")

# Gradient Boosting with optimal hyperparameters
log_output("\n--- Gradient Boosting Evaluation (Optimal Parameters) ---")
# Optimal params for this dataset size and dimensionality
gb_classifier = GradientBoostingClassifier(
    n_estimators=200,         # Good balance
    learning_rate=0.1,        # Standard rate
    max_depth=6,              # Moderate depth for high-dim data
    subsample=0.8,            # Reduce overfitting
    max_features='sqrt',      # Good for high-dimensional data
    random_state=42
)

gb_metrics, gb_dump = eval_sklearn_classifier(
    classifier=gb_classifier,
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    prefix="gb_", scale_features=False
)
log_output("Gradient Boosting Metrics:")
for k, v in gb_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")

# Logistic Regression with optimal hyperparameters
log_output("\n--- Logistic Regression Evaluation (Optimal Parameters) ---")
# Optimal params for high-dimensional features
lr_classifier = sk_LogisticRegression(
    C=0.1,                    # Some regularization for high-dim data
    penalty='l2',             # L2 regularization
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,            # Ensure convergence
    solver='liblinear',       # Good for high-dimensional data
    random_state=42
)

lr_metrics, lr_dump = eval_sklearn_classifier(
    classifier=lr_classifier,
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    prefix="lr_", scale_features=True
)
log_output("Logistic Regression Metrics:")
for k, v in lr_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")

# k-NN with optimal hyperparameters
log_output("\n--- k-NN Evaluation (Optimal Parameters) ---")
# Optimal params for this dataset size
knn_classifier = KNeighborsClassifier(
    n_neighbors=7,            # Good for ~4K samples
    weights='distance',       # Distance weighting
    metric='euclidean',       # Standard metric
    n_jobs=-1
)

knn_sklearn_metrics, knn_sklearn_dump = eval_sklearn_classifier(
    classifier=knn_classifier,
    train_feats=train_feats, train_labels=train_labels,
    test_feats=test_feats, test_labels=test_labels,
    prefix="knn_sklearn_", scale_features=True
)
log_output("k-NN (sklearn) Metrics:")
for k, v in knn_sklearn_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")

# ──────────────────────────── 11. Find Best Model ────────────────────────────
log_output("\n--- Finding Best Model by Accuracy ---")

# Collect all models and their accuracies
all_models = {
    "linear_probe": (linprobe_eval_metrics.get("lin_acc", 0), linprobe_dump),
    "knn_uni": (knn_eval_metrics.get("knn20_acc", 0), knn_dump),
    "protonet": (eval_metrics.get("acc", 0), {"preds_all": test_pred.cpu().numpy(), "targets_all": test_labels.cpu().numpy()}),
    "svm": (svm_metrics.get("svm_acc", 0), svm_dump),
    "random_forest": (rf_metrics.get("rf_acc", 0), rf_dump),
    "gradient_boosting": (gb_metrics.get("gb_acc", 0), gb_dump),
    "logistic_regression": (lr_metrics.get("lr_acc", 0), lr_dump),
    "knn_sklearn": (knn_sklearn_metrics.get("knn_sklearn_acc", 0), knn_sklearn_dump)
}

best_model_name = max(all_models.keys(), key=lambda x: all_models[x][0])
best_accuracy = all_models[best_model_name][0]
best_model_dump = all_models[best_model_name][1]

log_output(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Use best model predictions for further analysis
best_predictions = best_model_dump["preds_all"]
if isinstance(best_predictions, torch.Tensor):
    best_predictions = best_predictions.cpu().numpy()

# ──────────────────────────── 12. Top-k Patch Retrieval with Best Model ──────
log_output("\n--- Top-k Patch Retrieval using Best Model ---")

# Create a DataFrame from the test dataset paths and labels
test_imgs_df = pd.DataFrame(test_dataset.items, columns=['path', 'label'])
test_imgs_df.to_csv('test_imgs_df_mix_portal_tract.csv', index=False)

# Calculate confidence scores for ranking
if "probs_all" in best_model_dump and best_model_dump["probs_all"] is not None:
    probs = best_model_dump["probs_all"]
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    
    # For binary classification, use probability of positive class
    if probs.ndim == 2 and probs.shape[1] == 2:
        confidence_scores = probs[:, 1]  # Probability of class 1 (no response)
    else:
        confidence_scores = probs
else:
    # If no probabilities available, use predictions with random tie-breaking
    log_output("No probabilities available from best model, using prediction confidence")
    confidence_scores = best_predictions + np.random.rand(len(best_predictions)) * 0.1

# Get indices for each class sorted by confidence
class_0_indices = np.where(best_predictions == 0)[0]  # Response
class_1_indices = np.where(best_predictions == 1)[0]  # No Response

# Sort by confidence (higher confidence first)
class_0_sorted = class_0_indices[np.argsort(confidence_scores[class_0_indices])[::-1]]
class_1_sorted = class_1_indices[np.argsort(confidence_scores[class_1_indices])[::-1]]

# Import concat_images for concatenating images
from uni.downstream.utils import concat_images

# Function to save top-k images using best model
def save_topk_images_best_model(indices, class_name, k_values):
    for k in k_values:
        if len(indices) >= k:
            topk_indices = indices[:k]
            images = [Image.open(test_imgs_df['path'].iloc[idx]) for idx in topk_indices]
            concat_img = concat_images(images, gap=5)
            save_path = f"Top{k}_{class_name}_mix_portal_tract_best_model_{best_model_name}.png"
            concat_img.save(save_path)
            log_output(f"Saved top {k} {class_name} images to {save_path}")
            
            # Save indices to CSV
            df_indices = pd.DataFrame(topk_indices, columns=['index'])
            csv_path = f"top{k}_{class_name}_indices_mix_portal_tract_best_model_{best_model_name}.csv"
            df_indices.to_csv(csv_path, index=False)
            log_output(f"Saved top {k} {class_name} indices to {csv_path}")

# Save top-k images for both classes using best model
k_values = [5, 10, 15, 20, 25]
save_topk_images_best_model(class_0_sorted, "response", k_values)
save_topk_images_best_model(class_1_sorted, "no_response", k_values)

# ──────────────────────────── 13. Original ProtoNet Top-k Retrieval ─────────
log_output("\n--- Original ProtoNet Top-k Retrieval ---")

# Get top 100 queries for each class from the test features
dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=100)
log_output("Top 100 indices for each class computed.")

# --- Top-k No Response Test Samples (Class 1) ---
log_output("Top-k no response test samples for class 1")

# Retrieve top 100 indices for class 1
no_response_topk_inds = topk_inds[1]
# Save these indices to CSV
df_no_response = pd.DataFrame(no_response_topk_inds, columns=['index'])
csv_path_no_response = "topk_no_response_indices_mix_portal_tract.csv"
df_no_response.to_csv(csv_path_no_response, index=False)
log_output(f"Saved top-k indices for no response to {csv_path_no_response}")

# Generate concatenated images for different k values
for k in [5, 10, 15, 20, 25]:
    topk_no_response = no_response_topk_inds[:k]
    no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in topk_no_response], gap=5)
    save_path_no_response = f"Top{k}_no_response_mix_portal_tract_protonet.png"
    no_response_imgs.save(save_path_no_response)
    log_output(f"Saved concatenated top {k} no response image to {save_path_no_response}")

# --- Top-k Response Test Samples (Class 0) ---
log_output("Top-k Response test samples for class 0")

# Retrieve top 100 indices for class 0
response_topk_inds = topk_inds[0]
# Save these indices to CSV
df_response = pd.DataFrame(response_topk_inds, columns=['index'])
csv_path_response = "topk_response_indices_portal_tract.csv"
df_response.to_csv(csv_path_response, index=False)
log_output(f"Saved top-k indices for response to {csv_path_response}")

# Generate concatenated images for different k values
for k in [5, 10, 15, 20, 25]:
    topk_response = response_topk_inds[:k]
    response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in topk_response], gap=5)
    save_path_response = f"Top{k}_response_mix_portal_tract_protonet.png"
    response_imgs.save(save_path_response)
    log_output(f"Saved concatenated top {k} response image to {save_path_response}")

# ──────────────────────────── 14. Slide-level Evaluation with Best Model ─────
log_output("\n--- Slide-level Evaluation using Best Model ---")

# Dictionary to collect patch predictions for each slide
slide_preds = defaultdict(list)
slide_actual = {}

# Convert test labels to numpy if needed
test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels

# Iterate through the test dataset and corresponding patch predictions
for (img_path, actual_label), pred_label in zip(test_dataset.items, best_predictions):
    basename = os.path.basename(img_path)
    parts = basename.split('-')
    if len(parts) < 2:
        log_output(f"Filename format unexpected: {basename}")
        continue
    
    # The slide id is the biopsy patient id (second element)
    slide_id = parts[1]
    
    # Store prediction for this slide
    slide_preds[slide_id].append(pred_label)
    
    # Set the actual label for the slide
    if slide_id not in slide_actual:
        slide_actual[slide_id] = actual_label
    else:
        if slide_actual[slide_id] != actual_label:
            log_output(f"Warning: inconsistent actual labels for slide {slide_id}")

# Compute majority vote prediction for each slide
slide_pred_majority = {}
for slide, preds in slide_preds.items():
    # Majority vote: if average is less than 0.5 then label 0; otherwise label 1
    majority_label = 0 if np.mean(preds) < 0.5 else 1
    slide_pred_majority[slide] = majority_label

# Calculate slide-level accuracy
y_true_slides = []
y_pred_slides = []
for slide in slide_actual:
    y_true_slides.append(slide_actual[slide])
    y_pred_slides.append(slide_pred_majority.get(slide, 0))

slide_accuracy = np.mean(np.array(y_true_slides) == np.array(y_pred_slides))
slide_bacc = balanced_accuracy_score(y_true_slides, y_pred_slides)

log_output(f"\nSlide-level Evaluation Metrics (Best Model: {best_model_name}):")
log_output(f"Slide-level Accuracy: {slide_accuracy:.4f}")
log_output(f"Slide-level Balanced Accuracy: {slide_bacc:.4f}")

# Compute and log the confusion matrix
cm = confusion_matrix(y_true_slides, y_pred_slides)
log_output("Slide-level Confusion Matrix (rows: actual, columns: predicted):")
log_output(str(cm))

# Additional slide-level metrics
slide_metrics = get_eval_metrics(y_true_slides, y_pred_slides, get_report=True, prefix="slide_")
log_output("Detailed Slide-level Metrics:")
for k, v in slide_metrics.items():
    if "report" not in k:
        log_output(f"  {k}: {v:.4f}")
    else:
        log_output(f"  {k}: {v}")

# Save slide-level results
slide_results_df = pd.DataFrame({
    'slide_id': list(slide_actual.keys()),
    'actual_label': [slide_actual[sid] for sid in slide_actual.keys()],
    'predicted_label': [slide_pred_majority.get(sid, 0) for sid in slide_actual.keys()],
    'num_patches': [len(slide_preds[sid]) for sid in slide_actual.keys()]
})
slide_results_path = "slide_level_results_mix_portal_tract_best_model.csv"
slide_results_df.to_csv(slide_results_path, index=False)
log_output(f"Saved slide-level results to {slide_results_path}")

# ──────────────────────────── 15. Summary Comparison ─────────────────────────
log_output("\n--- Model Performance Summary ---")
log_output("Model Performance Comparison:")
for model_name, (accuracy, _) in all_models.items():
    log_output(f"{model_name}: {accuracy:.4f}")

log_output(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")
log_output(f"Analysis complete at {pd.Timestamp.now()}")

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
    'RAI Classification Biopsy #2': {'Response': 0, 'No Response': 1},
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

# Split clinical data with stratification
X_clinical_train, X_clinical_test, y_clinical_train, y_clinical_test, clinical_train_ids, clinical_test_ids = train_test_split(
    X_clinical_normalized, y_clinical, patient_ids, test_size=0.2, random_state=88, stratify=y_clinical
)

log_output(f"Clinical train data shape: {X_clinical_train.shape} ({len(clinical_train_ids)} patients)")
log_output(f"Clinical test data shape: {X_clinical_test.shape} ({len(clinical_test_ids)} patients)")

# Function to train and evaluate clinical models
def train_eval_sklearn_model(model, X_train, y_train, X_test, y_test, model_name, scale_features=False):
    """Train and evaluate a scikit-learn model on clinical data"""
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
    if y_prob is not None:
        if y_prob.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_prob[:, 1]) if len(np.unique(y_test)) > 1 else None
        else:  # Multi-class classification
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if len(np.unique(y_test)) > 1 else None
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

# ──────────────────────────── 11. Clinical Models Training ─────────────────────
log_output("\n=== Training Clinical Models ===")

# 1. Random Forest on Clinical Features
rf_clinical = RandomForestClassifier(random_state=777, criterion='gini', n_estimators=50, max_depth=4)
rf_clinical_results = train_eval_sklearn_model(
    rf_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "RandomForest (Clinical)", scale_features=False
)

# 2. SVM on Clinical Features
svm_clinical = SVC(C=1.0, kernel='rbf', gamma='auto', random_state=8888, probability=True)
svm_clinical_results = train_eval_sklearn_model(
    svm_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "SVM (Clinical)", scale_features=True
)

# 3. Gradient Boosting on Clinical Features
gb_clinical = GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, max_depth=4, random_state=2022)
gb_clinical_results = train_eval_sklearn_model(
    gb_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "GradientBoosting (Clinical)", scale_features=False
)

# 4. Logistic Regression on Clinical Features
lr_clinical = sk_LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=9999)
lr_clinical_results = train_eval_sklearn_model(
    lr_clinical, X_clinical_train, y_clinical_train,
    X_clinical_test, y_clinical_test, "LogisticRegression (Clinical)", scale_features=True
)

# ──────────────────────────── 15. Late Fusion with Dimension-Based Weights ────────
log_output("\n=== Dimension-Weighted Late Fusion ===")

# Extract slide-level probabilities from patch-level predictions
slide_probs = defaultdict(list)

# Collect probabilities for each slide from the best model
if "probs_all" in best_model_dump and best_model_dump["probs_all"] is not None:
    probs = best_model_dump["probs_all"]
    # Safely convert to numpy if it's a tensor
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    
    # For binary classification, use probability of positive class (No Response = class 1)
    if probs.ndim == 2 and probs.shape[1] == 2:
        for (img_path, _), prob in zip(test_dataset.items, probs):
            basename = os.path.basename(img_path)
            parts = basename.split('-')
            if len(parts) >= 2:
                slide_id = parts[1]  # Extract the patient ID
                slide_probs[slide_id].append(prob[1])  # Store prob of class 1
    else:
        log_output("Warning: Expected binary classification probabilities but got different format")
else:
    log_output("Warning: No probabilities available from best image model, using prediction confidence")
    # Fallback to using predictions with some uncertainty
    for (img_path, _), pred in zip(test_dataset.items, best_predictions):
        basename = os.path.basename(img_path)
        parts = basename.split('-')
        if len(parts) >= 2:
            slide_id = parts[1]
            slide_probs[slide_id].append(float(pred))

# Calculate average probability for each slide
slide_avg_probs = {}
for slide_id, probs in slide_probs.items():
    slide_avg_probs[slide_id] = np.mean(probs)

log_output(f"Computed slide-level probabilities for {len(slide_avg_probs)} slides")

# Prepare arrays for matched patients
common_patients = []
image_probs = []
clinical_probs = []
true_labels = []

# Calculate weights based on feature dimensions
img_dims = 1536  # Vision Transformer feature dimension
clinical_dims = 13  # Clinical feature dimension
total_dims = img_dims + clinical_dims
img_weight = img_dims / total_dims
clinical_weight = clinical_dims / total_dims

log_output(f"Dimension-based weights - Image: {img_weight:.4f}, Clinical: {clinical_weight:.4f}")

# Match clinical test patients with slides
for i, patient_id in enumerate(clinical_test_ids):
    if patient_id in slide_avg_probs:
        # This patient has both image and clinical data
        common_patients.append(patient_id)
        
        # Image model probability
        image_prob = slide_avg_probs[patient_id]
        image_probs.append(image_prob)
        
        # Get clinical model probabilities - average from all clinical models
        clinical_model_probs = []
        
        # Get RandomForest probability
        if hasattr(rf_clinical, "predict_proba"):
            X_test_sample = X_clinical_test[i].reshape(1, -1)
            rf_prob = rf_clinical.predict_proba(X_test_sample)[0][1]
            clinical_model_probs.append(rf_prob)
            
        # Get SVM probability - apply appropriate scaling
        if hasattr(svm_clinical, "predict_proba"):
            X_test_sample = X_clinical_test[i].reshape(1, -1)
            if svm_clinical_results['scaler'] is not None:
                X_test_sample = svm_clinical_results['scaler'].transform(X_test_sample)
            svm_prob = svm_clinical.predict_proba(X_test_sample)[0][1]
            clinical_model_probs.append(svm_prob)
            
        # Get Gradient Boosting probability
        if hasattr(gb_clinical, "predict_proba"):
            X_test_sample = X_clinical_test[i].reshape(1, -1)
            gb_prob = gb_clinical.predict_proba(X_test_sample)[0][1]
            clinical_model_probs.append(gb_prob)
            
        # Get Logistic Regression probability - apply appropriate scaling
        if hasattr(lr_clinical, "predict_proba"):
            X_test_sample = X_clinical_test[i].reshape(1, -1)
            if lr_clinical_results['scaler'] is not None:
                X_test_sample = lr_clinical_results['scaler'].transform(X_test_sample)
            lr_prob = lr_clinical.predict_proba(X_test_sample)[0][1]
            clinical_model_probs.append(lr_prob)
        
        # Average clinical probabilities
        avg_clinical_prob = np.mean(clinical_model_probs) if clinical_model_probs else 0.5
        clinical_probs.append(avg_clinical_prob)
        
        # Ground truth
        if isinstance(y_clinical_test, pd.Series):
            true_labels.append(y_clinical_test.iloc[i])  # Use iloc for integer-based position
        else:
            true_labels.append(y_clinical_test[i])

log_output(f"Found {len(common_patients)} patients with both image and clinical data")

# Perform dimension-weighted fusion
fused_probs = []
for img_p, clin_p in zip(image_probs, clinical_probs):
    # Weight according to feature dimensions
    weighted_prob = img_weight * img_p + clinical_weight * clin_p
    fused_probs.append(weighted_prob)

# Convert probabilities to predictions
fused_preds = [1 if p >= 0.5 else 0 for p in fused_probs]

# Evaluate fusion performance
fusion_acc = accuracy_score(true_labels, fused_preds)
fusion_bacc = balanced_accuracy_score(true_labels, fused_preds)

# Calculate AUC if possible
try:
    fusion_auc = roc_auc_score(true_labels, fused_probs)
    log_output(f"Dimension-weighted Late Fusion - AUC: {fusion_auc:.4f}")
except ValueError:
    log_output("Could not compute AUC (possibly only one class present)")
    fusion_auc = None

log_output(f"Dimension-weighted Late Fusion - Accuracy: {fusion_acc:.4f}")
log_output(f"Dimension-weighted Late Fusion - Balanced Accuracy: {fusion_bacc:.4f}")

# Compare with individual modalities
image_preds = [1 if p >= 0.5 else 0 for p in image_probs]
clinical_preds = [1 if p >= 0.5 else 0 for p in clinical_probs]

image_acc = accuracy_score(true_labels, image_preds)
image_bacc = balanced_accuracy_score(true_labels, image_preds)
clinical_acc = accuracy_score(true_labels, clinical_preds)
clinical_bacc = balanced_accuracy_score(true_labels, clinical_preds)

log_output("\n=== Comparison between modalities (matched patients) ===")
log_output(f"Image-only - Accuracy: {image_acc:.4f}, Balanced Accuracy: {image_bacc:.4f}")
log_output(f"Clinical-only - Accuracy: {clinical_acc:.4f}, Balanced Accuracy: {clinical_bacc:.4f}")
log_output(f"Dimension-weighted Fusion - Accuracy: {fusion_acc:.4f}, Balanced Accuracy: {fusion_bacc:.4f}")

# Save fusion results
fusion_results_df = pd.DataFrame({
    'patient_id': common_patients,
    'true_label': true_labels,
    'image_prob': image_probs,
    'clinical_prob': clinical_probs,
    'fused_prob': fused_probs,
    'fused_pred': fused_preds,
    'image_pred': image_preds,
    'clinical_pred': clinical_preds
})

fusion_results_path = "dimension_weighted_fusion_results.csv"
fusion_results_df.to_csv(fusion_results_path, index=False)
log_output(f"Saved fusion results to {fusion_results_path}")

# Calculate standard deviations using bootstrap resampling
import numpy as np
from sklearn.metrics import accuracy_score
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Bootstrap parameters
n_bootstraps = 1000
n_samples = len(true_labels)

# Arrays to store accuracy values for each bootstrap sample
pathology_accuracies = []
clinical_accuracies = []
fusion_accuracies = []

for _ in range(n_bootstraps):
    # Generate bootstrap indices with replacement
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    
    # Get bootstrap samples
    boot_true_labels = [true_labels[i] for i in indices]
    boot_image_probs = [image_probs[i] for i in indices]
    boot_clinical_probs = [clinical_probs[i] for i in indices]
    
    # Pathology-only accuracy
    boot_image_preds = [1 if p >= 0.5 else 0 for p in boot_image_probs]
    boot_image_acc = accuracy_score(boot_true_labels, boot_image_preds)
    pathology_accuracies.append(boot_image_acc)
    
    # Clinical-only accuracy
    boot_clinical_preds = [1 if p >= 0.5 else 0 for p in boot_clinical_probs]
    boot_clinical_acc = accuracy_score(boot_true_labels, boot_clinical_preds)
    clinical_accuracies.append(boot_clinical_acc)
    
    # Multimodal fusion accuracy
    boot_fused_probs = []
    for img_p, clin_p in zip(boot_image_probs, boot_clinical_probs):
        weighted_prob = img_weight * img_p + clinical_weight * clin_p
        boot_fused_probs.append(weighted_prob)
    
    boot_fused_preds = [1 if p >= 0.5 else 0 for p in boot_fused_probs]
    boot_fusion_acc = accuracy_score(boot_true_labels, boot_fused_preds)
    fusion_accuracies.append(boot_fusion_acc)

# Calculate standard deviation and mean for each approach
pathology_std = np.std(pathology_accuracies)
pathology_mean = np.mean(pathology_accuracies)

clinical_std = np.std(clinical_accuracies)
clinical_mean = np.mean(clinical_accuracies)

fusion_std = np.std(fusion_accuracies)
fusion_mean = np.mean(fusion_accuracies)

# Log results
log_output("\n=== Standard Deviation of Accuracy (Bootstrap Method) ===")
log_output(f"Pathology-only: Mean Accuracy = {pathology_mean:.4f}, Std Dev = {pathology_std:.4f}")
log_output(f"Clinical-only: Mean Accuracy = {clinical_mean:.4f}, Std Dev = {clinical_std:.4f}")
log_output(f"Multimodal Fusion: Mean Accuracy = {fusion_mean:.4f}, Std Dev = {fusion_std:.4f}")