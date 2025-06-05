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
from utils.my_utils import extract_features, log_output, get_eval_metrics, eval_sklearn_classifier

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
import umap.umap_ as umap
import tqdm

# Set seed for reproducibility
seed_value = 42
SEED=42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
response_path = "patches_mix_response_portal_tract"
no_response_path = "patches_mix_no_response_portal_tract"

# Collect all image file paths and their respective labels
seed_value = 42               # reproducible shuffling
batch_size = 16

# Shuffle and split into train and test sets
def collect_images(folder, label):
    return [(os.path.join(folder, f), label)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

response_images = collect_images(response_path, 0)
no_response_images = collect_images(no_response_path, 1)

def make_df(image_list):
    """Convert list[(path, label)] → DataFrame."""
    img_paths = [p for p, _ in image_list]
    labels    = [l for _, l in image_list]
    classes   = ["Response" if l == 0 else "No Response" for l in labels]
    return pd.DataFrame({
        "image_path": img_paths,
        "label":      labels,          # 0 or 1
        "class_name": classes          # human‑readable
    })

df_response = make_df(response_images)
df_response.to_csv('response_images.csv', index=False)
log_output(f"Response images saved to response_images.csv with {len(df_response)} entries.")
df_no_response = make_df(no_response_images)
df_no_response.to_csv('no_response_images.csv', index=False)
log_output(f"No Response images saved to no_response_images.csv with {len(df_no_response)} entries.")


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data      = data          # list of (filepath, label)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load encoder & transform
from uni import get_encoder
model, transform = get_encoder(enc_name="uni2-h", device=device)

# Build datasets/dataloaders
response_dataset = CustomImageDataset(response_images, transform=transform)
no_response_dataset = CustomImageDataset(no_response_images, transform=transform)

train_dataloader = torch.utils.data.DataLoader(response_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_dataloader  = torch.utils.data.DataLoader(no_response_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4)

# Extract features
response_features = extract_features(model, train_dataloader)
no_response_features = extract_features(model, test_dataloader)

# Save response_features and no_response_features to files
with open('response_features_portal_tract.pkl', 'wb') as f:
    pickle.dump(response_features, f)

with open('no_response_features_portal_tract.pkl', 'wb') as f:
    pickle.dump(no_response_features, f)

# Load features from files
# with open('response_features_portal_tract.pkl', 'rb') as f:
#     response_features = pickle.load(f)
# with open('no_response_features_portal_tract.pkl', 'rb') as f:
#     no_response_features = pickle.load(f)

log_output("Clinical Data Processing ...")

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

df_cleaned[X_clinical.columns] = X_clinical_normalized

with open('clinical_features_portal_tract.pkl', 'wb') as f:
    pickle.dump(df_cleaned, f)
log_output("Clinical Data Processing Completed.")