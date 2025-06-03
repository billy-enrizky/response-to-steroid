import torch
import torchvision
import os
import pickle
import random
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats # Added for p-value calculation

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
import umap.umap_ as umap

# ── 0. Global Settings ─────────────────────────────────────────────────
SEED = 42 # For reproducibility
K_TOP_PATCHES = 5 # As per feedback for question 6
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. Load Data ───────────────────────────────────────────────────────
# Load train_features and test_features from files
# IMPORTANT ASSUMPTION: We assume these pickle files also contain 'image_paths'
# corresponding to the embeddings and labels. If not, this part needs adjustment
# to correctly map embeddings to patient IDs.

try:
    with open('train_features_portal_tract_separated.pkl', 'rb') as f:
        loaded_train_features = pickle.load(f)

    with open('test_features_portal_tract_separated.pkl', 'rb') as f:
        loaded_test_features = pickle.load(f)
except FileNotFoundError:
    print("Error: train_features_portal_tract_separated.pkl or test_features_portal_tract_separated.pkl not found.")
    print("Please ensure these files are in the same directory as the script or provide the correct path.")
    print("For now, creating dummy data to proceed with the script structure.")
    # Dummy data to allow script to run for demonstration if files are missing
    def create_dummy_features(num_samples, num_features, num_paths):
        return {
            'embeddings': np.random.rand(num_samples, num_features).astype(np.float32),
            'labels': np.random.randint(0, 2, num_samples).astype(np.int64),
            'image_paths': [f'/dummy_path/patientA_patch{i}.jpg' for i in range(num_paths)] + \
                           [f'/dummy_path/patientB_patch{i}.jpg' for i in range(num_samples - num_paths)]
        }
    loaded_train_features = create_dummy_features(100, 64, 50) # 100 patches, 64 features
    loaded_test_features = create_dummy_features(50, 64, 25)   # 50 patches, 64 features


# Convert to tensors/numpy arrays
train_feats_orig = torch.Tensor(loaded_train_features['embeddings']).numpy() # Work with numpy for sklearn compatibility
train_labels_orig = torch.Tensor(loaded_train_features['labels']).type(torch.long).numpy()

test_feats_orig = torch.Tensor(loaded_test_features['embeddings']).numpy()
test_labels_orig = torch.Tensor(loaded_test_features['labels']).type(torch.long).numpy()


all_patient_csv_path = 'index_path_separated_portal_tract.csv'
if os.path.exists(all_patient_csv_path):
    all_patient_df_info = pd.read_csv(all_patient_csv_path)
else:
    print(f"Warning: {all_patient_csv_path} not found. Patient ID extraction might be limited or use dummy data.")
    # Create a dummy all_patient_df_info if CSV is not found, based on loaded features
    all_dummy_paths = train_image_paths_orig + test_image_paths_orig
    all_dummy_labels = np.concatenate([train_labels_orig, test_labels_orig])
    all_patient_df_info = pd.DataFrame({
        'image_path': all_dummy_paths,
        'label': all_dummy_labels,
        'class_name': ['Response' if lbl == 0 else 'No Response' for lbl in all_dummy_labels]
    })
train_image_paths_orig = all_patient_df_info['image_path']
test_image_paths_orig = all_patient_df_info['image_path']

# ── 2. Helper Functions ────────────────────────────────────────────────
def extract_patient_id_from_path(image_path):
    """Extracts patient ID from an image path string."""
    if image_path is None:
        return "unknown_patient"
    filename = os.path.basename(str(image_path))
    # Example: "response-5585_0087-78411-14619.jpg" -> "response-5585"
    # Example: "NR-1052_001.jpg" -> "NR-1052"
    parts = filename.split('-')
    if parts:
        # More robust extraction, handles cases like "response-5585" or "NR-1052"
        patient_id_candidate = parts[1]

    return filename.split('.')[0] # Fallback: use filename without extension

def create_patient_dataframe(image_paths, labels, features):
    """Creates a DataFrame with patient IDs, labels, and feature indices."""
    patient_ids = [extract_patient_id_from_path(p) for p in image_paths]
    df = pd.DataFrame({
        'image_path': image_paths,
        'patient_id': patient_ids,
        'label': labels,
        'feature_idx': range(len(labels)) # Original index in the features array
    })
    return df

train_df_orig = create_patient_dataframe(train_image_paths_orig, train_labels_orig, train_feats_orig)
test_df_orig = create_patient_dataframe(test_image_paths_orig, test_labels_orig, test_feats_orig)

COLORS = {0: "#1f77b4",    # blue   – Response
          1: "#d62728"}    # red    – No‑Response
CLASS_NAMES = {0: "Response", 1: "No-Response"}


def embed_2d(emb, method="umap", pca_dim=50, random_state=SEED):
    """Returns 2-D embedding using PCA→(t‑SNE | UMAP) or direct PCA."""
    if emb.shape[0] < 2 : # Not enough samples for embedding
        # Return a zero array of the correct shape if not enough samples
        return np.zeros((emb.shape[0], 2))
    if emb.shape[1] < 2: # Not enough features
        if emb.shape[1] == 1: # if only one feature, duplicate it for 2D
             return np.hstack([emb, emb])
        return np.zeros((emb.shape[0], 2))


    emb_std = normalize(emb, norm="l2")
    actual_pca_dim = min(pca_dim, emb_std.shape[0], emb_std.shape[1])
    if actual_pca_dim < 2 and method != "pca2": # PCA dim must be at least 2 for UMAP/tSNE
        # If cannot reduce to actual_pca_dim, try to use fewer components if possible
        # Or, if method is umap/tsne and pca can't run meaningfully, fall back
        print(f"Warning: PCA dim ({actual_pca_dim}) too small for UMAP/tSNE with pca_dim={pca_dim}. Trying direct {method} or pca2.")
        if emb_std.shape[1] < 2: # Not enough features for any reduction
            return np.zeros((emb_std.shape[0], 2)) # Or handle as error
        # Fallback to pca2 if UMAP/TSNE pre-reduction is not possible
        method = "pca2"


    if method == "pca2":
        pca_n_components = min(2, emb_std.shape[0], emb_std.shape[1])
        if pca_n_components < 1: return np.zeros((emb_std.shape[0], 2))
        return PCA(n_components=pca_n_components, random_state=random_state).fit_transform(emb_std)

    # For UMAP/t-SNE, apply PCA first
    # Ensure pca_dim is not more than available samples or features
    pca_intermediate_dim = min(pca_dim, emb_std.shape[0] -1, emb_std.shape[1]-1) # -1 for some algorithms' requirements
    pca_intermediate_dim = max(2, pca_intermediate_dim) # must be at least 2

    if pca_intermediate_dim >= 2:
      emb_pca = PCA(n_components=pca_intermediate_dim, random_state=random_state).fit_transform(emb_std)
    else: # Not enough features/samples for intermediate PCA
      emb_pca = emb_std


    # Ensure n_neighbors for UMAP is less than sample size in emb_pca
    n_neighbors_umap = 15
    if emb_pca.shape[0] <= n_neighbors_umap:
        n_neighbors_umap = emb_pca.shape[0] - 1
    if n_neighbors_umap < 2 : # UMAP requires at least 2 neighbors
         print(f"Warning: Not enough samples for UMAP ({emb_pca.shape[0]}). Falling back to PCA for 2D.")
         pca_n_components = min(2, emb_std.shape[0], emb_std.shape[1])
         if pca_n_components < 1: return np.zeros((emb_std.shape[0], 2))
         return PCA(n_components=pca_n_components, random_state=random_state).fit_transform(emb_std)


    if method == "tsne":
        perplexity_tsne = 30
        if emb_pca.shape[0] -1 < perplexity_tsne:
            perplexity_tsne = max(5, emb_pca.shape[0] - 2) # Adjust perplexity
        if perplexity_tsne <5 : # TSNE might fail
             print(f"Warning: Not enough samples for t-SNE ({emb_pca.shape[0]}). Falling back to PCA for 2D.")
             pca_n_components = min(2, emb_std.shape[0], emb_std.shape[1])
             if pca_n_components < 1: return np.zeros((emb_std.shape[0], 2))
             return PCA(n_components=pca_n_components, random_state=random_state).fit_transform(emb_std)

        tsne = TSNE(n_components=2, perplexity=perplexity_tsne, metric="euclidean",
                    init="random", random_state=random_state, learning_rate='auto')
        return tsne.fit_transform(emb_pca)
    elif method == "umap":
        um = umap.UMAP(n_components=2, n_neighbors=n_neighbors_umap, min_dist=0.1,
                       metric="euclidean", random_state=random_state)
        return um.fit_transform(emb_pca)
    else:
        raise ValueError("method must be 'tsne', 'umap', or 'pca2'")

# ── 3. How many features each patient have? ───────────────────────────
def report_features_per_patient(df, feature_array, cohort_name):
    print(f"\n--- Q3: Features per Patient ({cohort_name}) ---")
    if df.empty or feature_array.ndim == 1 or feature_array.shape[0] == 0 : # feature_array could be 1D if only one feature selected
        print("No data to report.")
        return

    feature_dimensionality = feature_array.shape[1] if feature_array.ndim > 1 else 1
    patches_per_patient = df.groupby('patient_id').size()

    print(f"Feature dimensionality per patch: {feature_dimensionality}")
    print("Number of patches per patient:")
    for patient_id, count in patches_per_patient.items():
        print(f"  Patient {patient_id}: {count} patches")
    if patches_per_patient.empty:
        print("  No patients found in this cohort.")
    return patches_per_patient, feature_dimensionality

# ── 4. Scatter plots of the features ───────────────────────────────────
def plot_scatter_by_class(embeddings, labels, title_prefix, method="umap", save_fig_path=None):
    if embeddings.shape[0] == 0:
        print(f"Skipping scatter plot for {title_prefix}: No embeddings to plot.")
        return None
    print(f"\n--- Q4: Scatter Plot by Class ({title_prefix}) ---")
    emb2 = embed_2d(embeddings, method=method)
    if emb2.shape[0] == 0 :
        print(f"Could not generate 2D embedding for {title_prefix}")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_label in np.unique(labels):
        idx = labels == cls_label
        ax.scatter(emb2[idx, 0], emb2[idx, 1], s=15, alpha=0.8, c=COLORS.get(cls_label, "gray"),
                   label=f"{CLASS_NAMES.get(cls_label, f'Class {cls_label}')} ({idx.sum()})")
    ax.set_title(f"{method.upper()} of patch embeddings ({title_prefix}) - By Class")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
        print(f"✓ Class scatter plot saved to {save_fig_path}")
    plt.show()
    return emb2

def plot_scatter_by_patient(embeddings, patient_ids, labels, title_prefix, method="umap", save_fig_path=None):
    if embeddings.shape[0] == 0:
        print(f"Skipping scatter plot for {title_prefix}: No embeddings to plot.")
        return None
    print(f"\n--- Q4: Scatter Plot by Patient ({title_prefix}) ---")
    emb2 = embed_2d(embeddings, method=method)
    if emb2.shape[0] == 0 :
        print(f"Could not generate 2D embedding for {title_prefix}")
        return None

    df_plot = pd.DataFrame({
        'dim1': emb2[:, 0],
        'dim2': emb2[:, 1],
        'patient_id': patient_ids,
        'label': labels
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    unique_patients = df_plot['patient_id'].unique()
    # Use a broader color palette for patients if many, or cycle through a few
    patient_colors = plt.cm.get_cmap('tab20', len(unique_patients))

    for i, patient_id in enumerate(unique_patients):
        sub_df = df_plot[df_plot['patient_id'] == patient_id]
        # Use the class color for consistency in patient plots, or patient_colors(i) for distinct patient colors
        # For this example, let's use class color and vary markers or just plot.
        # For many patients, distinct colors are hard. Grouping by class color:
        patient_class_label = sub_df['label'].iloc[0]
        ax.scatter(sub_df['dim1'], sub_df['dim2'], s=20, alpha=0.7,
                   color=COLORS.get(patient_class_label, "grey"), # Color by class
                   label=f"{patient_id} ({CLASS_NAMES.get(patient_class_label)})" if len(unique_patients) < 15 else None) # Avoid too many legend entries

    ax.set_title(f"{method.upper()} of patch embeddings ({title_prefix}) - By Patient (colored by class)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if len(unique_patients) < 15 : # Show legend if not too cluttered
        ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        # Add a general legend for class colors if patient legend is hidden
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=CLASS_NAMES[lbl],
                              markerfacecolor=clr, markersize=10) for lbl, clr in COLORS.items()]
        ax.legend(handles=handles, title="Classes", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
        print(f"✓ Patient scatter plot saved to {save_fig_path}")
    plt.show()


# ── 5. Calculate p-values for each image feature ──────────────────────
def calculate_feature_p_values(features, labels, cohort_name):
    print(f"\n--- Q5: P-values for Image Features ({cohort_name}) ---")
    if features.ndim == 1 or features.shape[0] == 0 or features.shape[1] == 0:
        print("Not enough data or features to calculate p-values.")
        return pd.DataFrame()

    class0_feats = features[labels == 0]
    class1_feats = features[labels == 1]

    if class0_feats.shape[0] < 2 or class1_feats.shape[0] < 2:
        print("Not enough samples in one or both classes to calculate p-values.")
        return pd.DataFrame()

    p_values = []
    num_features = features.shape[1]
    for i in range(num_features):
        # Using Mann-Whitney U test as it's non-parametric
        try:
            stat, p_val = stats.mannwhitneyu(class0_feats[:, i], class1_feats[:, i], alternative='two-sided')
            p_values.append(p_val)
        except ValueError as e: # Handle cases like all values being the same
            # print(f"Could not calculate p-value for feature {i}: {e}")
            p_values.append(np.nan)


    p_values_df = pd.DataFrame({'feature_index': range(num_features), 'p_value': p_values})
    print(p_values_df.head())
    # print(f"Number of features with p-value < 0.05: {(p_values_df['p_value'] < 0.05).sum()}")
    return p_values_df

# ── 6. Top-k important patches identification ──────────────────────────
def get_top_k_patches_per_patient(features, df, k, train_class_centroids):
    print(f"\n--- Identifying Top-{k} Patches ---")
    top_k_indices_all_patients = []
    top_k_patient_ids = []
    top_k_labels = []

    # Normalize features for cosine distance calculation
    features_norm = normalize(features, norm="l2")

    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id]
        patient_labels = patient_data['label'].values
        if len(patient_labels) == 0:
            continue
        patient_true_label = patient_labels[0] # Assuming all patches for a patient have same label

        patient_feature_indices = patient_data['feature_idx'].values
        patient_features_norm = features_norm[patient_feature_indices]

        if patient_features_norm.shape[0] == 0:
            continue

        # Get the appropriate class centroid
        class_centroid_norm = train_class_centroids[patient_true_label]

        # Calculate cosine distances from patient's patches to their class centroid
        # Cosine distance = 1 - cosine similarity
        distances = pairwise_distances(patient_features_norm, class_centroid_norm.reshape(1, -1), metric='cosine').flatten()

        # Select top k patches (those with smallest distance)
        num_patches_to_select = min(k, len(distances))
        if num_patches_to_select > 0 :
            top_k_local_indices = np.argsort(distances)[:num_patches_to_select]
            top_k_global_indices = patient_feature_indices[top_k_local_indices]

            top_k_indices_all_patients.extend(top_k_global_indices)
            top_k_patient_ids.extend([patient_id] * len(top_k_global_indices))
            top_k_labels.extend([patient_true_label] * len(top_k_global_indices))
        else:
            print(f"Patient {patient_id} had no patches after filtering or 0 patches to select.")


    if not top_k_indices_all_patients:
        print("Warning: No top-k patches were selected. Subsequent analysis might fail or be empty.")
        return np.array([]), pd.DataFrame(columns=['patient_id', 'label', 'feature_idx']), np.array([])


    # Create new features and DataFrame for top-k patches
    top_k_features = features[top_k_indices_all_patients]
    new_df = pd.DataFrame({
        'patient_id': top_k_patient_ids,
        'label': top_k_labels,
        # The feature_idx here is the new index within the top_k_features array
        'feature_idx': range(len(top_k_indices_all_patients)),
        'original_feature_idx': top_k_indices_all_patients # Keep track of original index if needed
    })
    return top_k_features, new_df, np.array(top_k_labels)


# Wrapper function to perform analysis
def perform_analysis_suite(features, df, labels, cohort_name_prefix, train_class_centroids_for_top_k=None, is_top_k_run=False):
    print(f"\n{'='*20} ANALYSIS SUITE: {cohort_name_prefix} {'='*20}")

    # Q3: Features per patient
    report_features_per_patient(df, features, cohort_name_prefix)

    # Q4: Scatter plots
    # By Class
    plot_scatter_by_class(features, labels,
                          title_prefix=f"{cohort_name_prefix} - UMAP", method="umap",
                          save_fig_path=f"{cohort_name_prefix.lower().replace(' ', '_')}_umap_by_class.png")
    # By Patient
    plot_scatter_by_patient(features, df['patient_id'].values, labels,
                            title_prefix=f"{cohort_name_prefix} - UMAP", method="umap",
                            save_fig_path=f"{cohort_name_prefix.lower().replace(' ', '_')}_umap_by_patient.png")

    # Q5: P-values
    calculate_feature_p_values(features, labels, cohort_name_prefix)

    if not is_top_k_run: # Only do top-k identification if it's the initial run
        print(f"\n--- Q6: Identifying Top-{K_TOP_PATCHES} Patches for {cohort_name_prefix} ---")
        # Calculate class centroids from THIS cohort's training data if it's the main training set
        # For test set, it should use centroids from the main training set.
        # For simplicity in this script, if train_class_centroids_for_top_k is not given, calculate from current `features`
        # This is okay for the initial train run. For test, it must be passed.
        current_centroids = {}
        if train_class_centroids_for_top_k is None: # Calculate if this is the primary training set
            print("Calculating class centroids for top-k patch selection...")
            for cls_label in np.unique(labels):
                cls_feats = features[labels == cls_label]
                if cls_feats.shape[0] > 0:
                     current_centroids[cls_label] = normalize(cls_feats.mean(axis=0).reshape(1, -1), norm="l2")[0]
                else: # Handle case where a class might be empty
                    current_centroids[cls_label] = np.zeros(features.shape[1]) # Placeholder
            centroids_to_use = current_centroids
            # Store these to pass to test set if this is the train run
            if "Train" in cohort_name_prefix:
                global global_train_class_centroids
                global_train_class_centroids = centroids_to_use

        else: # Use provided centroids (e.g. from training set for test set analysis)
            centroids_to_use = train_class_centroids_for_top_k

        if not centroids_to_use or not any(c.any() for c in centroids_to_use.values()):
            print(f"Error: Class centroids are not properly computed or are all zero for {cohort_name_prefix}. Skipping top-k analysis.")
            return

        features_topk, df_topk, labels_topk = get_top_k_patches_per_patient(
            features, df, K_TOP_PATCHES, centroids_to_use
        )

        if features_topk.shape[0] > 0:
            print(f"\n--- Redoing Analysis for Top-{K_TOP_PATCHES} Patches ({cohort_name_prefix}) ---")
            perform_analysis_suite(features_topk, df_topk, labels_topk,
                                   f"{cohort_name_prefix} - Top {K_TOP_PATCHES} Patches",
                                   is_top_k_run=True) # Pass centroids if needed for consistency, but it's not used in is_top_k_run
        else:
            print(f"Skipping top-k analysis for {cohort_name_prefix} as no top-k patches were selected.")


# ── Execute Analysis ───────────────────────────────────────────────────
global_train_class_centroids = None # To store train centroids for test set top-k selection

print("\nProcessing Training Data (Initial)...")
perform_analysis_suite(train_feats_orig, train_df_orig, train_labels_orig, "Train Cohort")

print("\n\nProcessing Test Data (Initial)...")
if global_train_class_centroids is None:
    print("Warning: Training class centroids not calculated. Test set top-k selection might be suboptimal or fail if it relies on them implicitly.")
    # As a fallback, if train set was empty or failed, calculate test-specific centroids for its own top-k (less ideal)
    perform_analysis_suite(test_feats_orig, test_df_orig, test_labels_orig, "Test Cohort", train_class_centroids_for_top_k=None)
else:
    perform_analysis_suite(test_feats_orig, test_df_orig, test_labels_orig, "Test Cohort", train_class_centroids_for_top_k=global_train_class_centroids)

print("\n\n--- Analysis Complete ---")