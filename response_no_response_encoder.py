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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from utils.my_utils import extract_features, log_output, get_eval_metrics, eval_sklearn_classifier

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
# response_features = extract_features(model, train_dataloader)
# no_response_features = extract_features(model, test_dataloader)

# Save response_features and no_response_features to files
# with open('response_features_portal_tract.pkl', 'wb') as f:
#     pickle.dump(response_features, f)

# with open('no_response_features_portal_tract.pkl', 'wb') as f:
#     pickle.dump(no_response_features, f)

with open('response_features_portal_tract.pkl', 'rb') as f:
    response_features = pickle.load(f)
with open('no_response_features_portal_tract.pkl', 'rb') as f:
    no_response_features = pickle.load(f)

log_output("Visualizing random patches before and after encoding...")

# Select 10 random patches for visualization
def visualize_patches_and_embeddings(model, dataloader, transform, num_patches=10):
    """Visualize random patches before and after encoding"""
    
    # Collect some sample images and their embeddings
    sample_images = []
    sample_embeddings = []
    sample_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(dataloader):
            if len(sample_images) >= num_patches:
                break
                
            batch = batch.to(device)
            embeddings = model(batch).detach().cpu()
            
            # Add samples from this batch
            for i in range(batch.size(0)):
                if len(sample_images) >= num_patches:
                    break
                sample_images.append(batch[i].cpu())
                sample_embeddings.append(embeddings[i])
                sample_labels.append(labels[i].item())
    
    # Create visualization
    fig, axes = plt.subplots(3, num_patches, figsize=(20, 8))
    
    for i in range(num_patches):
        # Original image (denormalize for display)
        img = sample_images[i]
        # Denormalize the image (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Display original image
        axes[0, i].imshow(img_denorm.permute(1, 2, 0))
        axes[0, i].set_title(f'Patch {i+1}\nLabel: {"Response" if sample_labels[i] == 0 else "No Response"}')
        axes[0, i].axis('off')
        
        # Display embedding as heatmap (first 64 dimensions reshaped to 8x8)
        embedding = sample_embeddings[i][:64].reshape(8, 8)
        im1 = axes[1, i].imshow(embedding, cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Embedding Heatmap\n(First 64 dims)')
        axes[1, i].axis('off')
        
        # Display embedding distribution
        axes[2, i].hist(sample_embeddings[i].numpy(), bins=30, alpha=0.7, color='blue')
        axes[2, i].set_title(f'Embedding Distribution\n(Mean: {sample_embeddings[i].mean():.3f})')
        axes[2, i].set_xlabel('Value')
        axes[2, i].set_ylabel('Frequency')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Original\nPatches', transform=axes[0, 0].transAxes, 
                    rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Embedding\nHeatmaps', transform=axes[1, 0].transAxes, 
                    rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
    axes[2, 0].text(-0.1, 0.5, 'Embedding\nDistributions', transform=axes[2, 0].transAxes, 
                    rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('patch_encoding_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print embedding statistics
    log_output("\nEmbedding Statistics:")
    for i in range(num_patches):
        emb = sample_embeddings[i]
        log_output(f"Patch {i+1} - Shape: {emb.shape}, Mean: {emb.mean():.4f}, "
                  f"Std: {emb.std():.4f}, Min: {emb.min():.4f}, Max: {emb.max():.4f}")

# Visualize patches from both response and no_response datasets
log_output("Visualizing Response patches:")
visualize_patches_and_embeddings(model, train_dataloader, transform, num_patches=5)

log_output("Visualizing No Response patches:")
visualize_patches_and_embeddings(model, test_dataloader, transform, num_patches=5)

# Additional visualization: t-SNE of embeddings
from sklearn.manifold import TSNE

def visualize_embedding_space(response_features, no_response_features, n_samples=500):
    """Visualize embedding space using t-SNE"""
    
    # Sample embeddings for visualization
    resp_emb = response_features['embeddings']
    no_resp_emb = no_response_features['embeddings']
    
    # Sample random indices
    resp_indices = np.random.choice(len(resp_emb), min(n_samples//2, len(resp_emb)), replace=False)
    no_resp_indices = np.random.choice(len(no_resp_emb), min(n_samples//2, len(no_resp_emb)), replace=False)
    
    # Combine embeddings and labels
    embeddings_sample = np.vstack([
        resp_emb[resp_indices],
        no_resp_emb[no_resp_indices]
    ])
    labels_sample = np.concatenate([
        np.zeros(len(resp_indices)),
        np.ones(len(no_resp_indices))
    ])
    
    # Apply t-SNE
    log_output(f"Applying t-SNE to {len(embeddings_sample)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_sample, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ticks=[0, 1], label='Response (0) / No Response (1)')
    plt.title('t-SNE Visualization of Vision Transformer Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

log_output("Creating t-SNE visualization of embedding space...")
visualize_embedding_space(response_features, no_response_features)

log_output("Visualization completed. Check 'patch_encoding_visualization.png' and 'embedding_tsne_visualization.png'")
exit(0)
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