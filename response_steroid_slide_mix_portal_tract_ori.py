import os
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

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

csv_out = Path("/cluster/projects/bhatgroup/response_to_steroid/index_path_mix_portal_tract.csv")
pd.concat([make_df(train_data), make_df(test_data)]).to_csv(csv_out, index_label="idx")
print(f"[INFO] saved slide index → {csv_out}")

# Extract features
# from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
# train_features = extract_patch_features_from_dataloader(model, train_dataloader)

# # Save train_features and test_features to files
# with open('/cluster/projects/bhatgroup/response_to_steroid/train_features_mix_portal_tract.pkl', 'wb') as f:
#     pickle.dump(train_features, f)

# test_features = extract_patch_features_from_dataloader(model, test_dataloader)
# with open('/cluster/projects/bhatgroup/response_to_steroid/test_features_mix_portal_tract.pkl', 'wb') as f:
#     pickle.dump(test_features, f)

# Load train_features and test_features from files
with open('/cluster/projects/bhatgroup/response_to_steroid/train_features_mix_portal_tract.pkl', 'rb') as f:
    loaded_train_features = pickle.load(f)

with open('/cluster/projects/bhatgroup/response_to_steroid/test_features_mix_portal_tract.pkl', 'rb') as f:
    loaded_test_features = pickle.load(f)

# Convert to tensors
train_feats = torch.Tensor(loaded_train_features['embeddings'])
train_labels = torch.Tensor(loaded_train_features['labels']).type(torch.long)
test_feats = torch.Tensor(loaded_test_features['embeddings'])
test_labels = torch.Tensor(loaded_test_features['labels']).type(torch.long)

# Setup logging
log_file = "/cluster/projects/bhatgroup/response_to_steroid/model_evaluation_mix_portal_tract.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

log_output(f"\n\n\nTime now is: {pd.Timestamp.now()}\n\n\n")

# Evaluation methods remain unchanged
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics

# Linear probe evaluation
linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
    train_feats=train_feats,
    train_labels=train_labels,
    valid_feats=None,
    valid_labels=None,
    test_feats=test_feats,
    test_labels=test_labels,
    max_iter=1000,
    verbose=True,
)
log_output("\n\n\n Here is the Linear Probe Evaluation Metrics \n\n\n")
log_output(str(linprobe_eval_metrics))

# KNN evaluation
knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
    train_feats=train_feats,
    train_labels=train_labels,
    test_feats=test_feats,
    test_labels=test_labels,
    center_feats=True,
    normalize_feats=True,
    n_neighbors=20
)
log_output("\n\n\n Here is the KNN Evaluation Metrics \n\n\n")
log_output(str(knn_eval_metrics))

log_output("\n\n\n Here is the ProtoNet Evaluation Metrics \n\n\n")
log_output(str(proto_eval_metrics))

# Few-shot evaluation
fewshot_episodes, fewshot_dump = eval_fewshot(
    train_feats=train_feats,
    train_labels=train_labels,
    test_feats=test_feats,
    test_labels=test_labels,
    n_iter=500,
    n_way=2,
    n_shot=4,
    n_query=test_feats.shape[0],
    center_feats=True,
    normalize_feats=True,
    average_feats=True,
)
log_output("\n\n\n Here is the Few Shot Episodes \n\n\n")
log_output(str(fewshot_episodes))
log_output("\n\n\n Here is the Few Shot Dump \n\n\n")
log_output(str(fewshot_dump))

# ProtoNet evaluation
proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
proto_clf.fit(train_feats, train_labels, verbose=True)
log_output('\n\n\n What our prototypes look like\n\n\n')
log_output(str(proto_clf.prototype_embeddings.shape))

test_pred = proto_clf.predict(test_feats)
eval_metrics = get_eval_metrics(test_labels, test_pred, get_report=False)
log_output("\n\n\n ProtoNet Evaluation Metrics \n\n\n")
log_output(str(eval_metrics))


# ----- Additional Evaluation: Top-k Retrieval for Patches -----
# Print and log the mapping from class names to indices
#print("train_dataset.class_to_idx:", train_dataset.__dict__.get('class_to_idx', 'Not Defined'))
#log_output("train_dataset.class_to_idx: " + str(train_dataset.__dict__.get('class_to_idx', 'Not Defined')))

# Get top 100 queries for each class from the test features
dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=100)
log_output("Top 100 indices for each class computed.")

# Create a DataFrame from the test dataset paths and labels.
# Our CustomImageDataset stores the data in 'data'
test_imgs_df = pd.DataFrame(test_dataset.items, columns=['path', 'label'])
test_imgs_df.to_csv('/cluster/projects/bhatgroup/response_to_steroid/test_imgs_df_mix_portal_tract.csv', index=False)


# Import concat_images for concatenating images
from uni.downstream.utils import concat_images

# For demonstration, we assume:
# - Class 1 corresponds to "no response" (as per the instructions) and
# - Class 0 corresponds to "Response".
# (Check train_dataset.class_to_idx output for verification)

# --- Top-k No Response Test Samples (Class 1) ---
print("Top-k no response test samples")
log_output("Top-k no response test samples for class 1")

# Retrieve top 100 indices for class 1
no_response_topk_inds = topk_inds[1]
# Save these indices to CSV
df_no_response = pd.DataFrame(no_response_topk_inds, columns=['index'])
csv_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/topk_no_response_indices_mix_portal_tract.csv"
df_no_response.to_csv(csv_path_no_response, index=False)
log_output(f"Saved top-k indices for no response to {csv_path_no_response}")

# Select top 5 indices for concatenation
top5_no_response = no_response_topk_inds[:5]
# Concatenate corresponding images
no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top5_no_response],   gap=5)
# Save the concatenated image
save_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/Top5_no_response_mix_portal_tract.png"
no_response_imgs.save(save_path_no_response)
log_output(f"Saved concatenated top 5 no response image to {save_path_no_response}")

# Select top 10 indices for concatenation
top10_no_response = no_response_topk_inds[:10]
# Concatenate corresponding images
no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top10_no_response],   gap=5)
# Save the concatenated image
save_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/Top10_no_response_mix_portal_tract.png"
no_response_imgs.save(save_path_no_response)
log_output(f"Saved concatenated top 10 no response image to {save_path_no_response}")

# Select top 15 indices for concatenation
top15_no_response = no_response_topk_inds[:15]
# Concatenate corresponding images
no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top15_no_response],   gap=5)
# Save the concatenated image
save_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/Top15_no_response_mix_portal_tract.png"
no_response_imgs.save(save_path_no_response)
log_output(f"Saved concatenated top 15 no response image to {save_path_no_response}")


# Select top 20 indices for concatenation
top20_no_response = no_response_topk_inds[:20]
# Concatenate corresponding images
no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top20_no_response],   gap=5)
# Save the concatenated image
save_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/Top20_no_response_mix_portal_tract.png"
no_response_imgs.save(save_path_no_response)
log_output(f"Saved concatenated top 20 no response image to {save_path_no_response}")

# Select top 25 indices for concatenation
top25_no_response = no_response_topk_inds[:25]
# Concatenate corresponding images
no_response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top25_no_response],   gap=5)
# Save the concatenated image
save_path_no_response = "/cluster/projects/bhatgroup/response_to_steroid/Top25_no_response_mix_portal_tract.png"
no_response_imgs.save(save_path_no_response)
log_output(f"Saved concatenated top 25 no response image to {save_path_no_response}")


# --- Top-k Response Test Samples (Class 0) ---
print("Top-k Response test samples")
log_output("Top-k Response test samples for class 0")

# Retrieve top 100 indices for class 0
response_topk_inds = topk_inds[0]
# Save these indices to CSV
df_response = pd.DataFrame(response_topk_inds, columns=['index'])
csv_path_response = "/cluster/projects/bhatgroup/response_to_steroid/topk_response_indices_portal_tract.csv"
df_response.to_csv(csv_path_response, index=False)
log_output(f"Saved top-k indices for response to {csv_path_response}")

# Select top 5 indices for concatenation
top5_response = response_topk_inds[:5]
# Concatenate corresponding images
response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top5_response],   gap=5)
# Save the concatenated image
save_path_response = "/cluster/projects/bhatgroup/response_to_steroid/Top5_response_mix_portal_tract.png"
response_imgs.save(save_path_response)
log_output(f"Saved concatenated top 5 response image to {save_path_response}")


# Select top 10 indices for concatenation
top10_response = response_topk_inds[:10]
# Concatenate corresponding images
response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top10_response],   gap=5)
# Save the concatenated image
save_path_response = "/cluster/projects/bhatgroup/response_to_steroid/Top10_response_mix_portal_tract.png"
response_imgs.save(save_path_response)
log_output(f"Saved concatenated top 10 response image to {save_path_response}")

# Select top 15 indices for concatenation
top15_response = response_topk_inds[:15]
# Concatenate corresponding images
response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top15_response],   gap=5)
# Save the concatenated image
save_path_response = "/cluster/projects/bhatgroup/response_to_steroid/Top15_response_mix_portal_tract.png"
response_imgs.save(save_path_response)
log_output(f"Saved concatenated top 15 response image to {save_path_response}")

# Select top 20 indices for concatenation
top20_response = response_topk_inds[:20]
# Concatenate corresponding images
response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top20_response],   gap=5)
# Save the concatenated image
save_path_response = "/cluster/projects/bhatgroup/response_to_steroid/Top20_response_mix_portal_tract.png"
response_imgs.save(save_path_response)
log_output(f"Saved concatenated top 20 response image to {save_path_response}")

# Select top 25 indices for concatenation
top25_response = response_topk_inds[:25]
# Concatenate corresponding images
response_imgs = concat_images([Image.open(test_imgs_df['path'].iloc[idx]) for idx in top25_response],   gap=5)
# Save the concatenated image
save_path_response = "/cluster/projects/bhatgroup/response_to_steroid/Top25_response_mix_portal_tract.png"
response_imgs.save(save_path_response)
log_output(f"Saved concatenated top 25 response image to {save_path_response}")

# ----- Slide-level evaluation -----
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# Dictionary to collect patch predictions for each slide (biopsy patient id)
slide_preds = defaultdict(list)
slide_actual = {}

# Convert predicted tensor to numpy array of ints
pred_labels = test_pred.cpu().numpy().astype(int)

# Iterate through the test dataset and corresponding patch predictions
# Note: test_dataset.data order is assumed to be the same as test_feats order.
for (img_path, _), pred in zip(test_dataset.items, pred_labels):
    basename = os.path.basename(img_path)
    parts = basename.split('-')
    if len(parts) < 2:
        log_output(f"Filename format unexpected: {basename}")
        continue
    # The slide id is the biopsy patient id (second element)
    slide_id = parts[1]
    # Actual slide label: first element (e.g., "response" or "no_response")
    actual_str = parts[0].lower()  # make it lowercase for consistency
    actual_label = 0 if ("response" in actual_str and "no" not in actual_str) else 1
    slide_preds[slide_id].append(pred)
    # Set the actual label for the slide (assuming consistency across patches)
    if slide_id not in slide_actual:
        slide_actual[slide_id] = actual_label
    else:
        if slide_actual[slide_id] != actual_label:
            log_output(f"Warning: inconsistent actual labels for slide {slide_id}")

# Compute majority vote prediction for each slide
slide_pred_majority = {}
for slide, preds in slide_preds.items():
    # Majority vote: if average is less than 0.5 then label 0; otherwise label 1.
    majority_label = 0 if np.mean(preds) < 0.5 else 1
    slide_pred_majority[slide] = majority_label

# Calculate slide-level accuracy
y_true = []
y_pred = []
for slide in slide_actual:
    y_true.append(slide_actual[slide])
    y_pred.append(slide_pred_majority.get(slide, 0))  # default to 0 if slide missing

slide_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
log_output("\n\n\n Slide-level Evaluation Metrics \n\n\n")
log_output(f"Slide-level Accuracy: {slide_accuracy:.4f}")

# Compute and log the confusion matrix
cm = confusion_matrix(y_true, y_pred)
log_output("Confusion Matrix (rows: actual, columns: predicted):")
log_output(str(cm))
