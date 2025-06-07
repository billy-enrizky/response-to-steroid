import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from PIL import Image
import seaborn as sns
import shutil
from pathlib import Path

# Set up logging
log_file = "top_patches_visualization.log"
def log_output(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

def load_data():
    """Load model results and patch embeddings"""
    log_output("Loading results and feature data...")
    
    # Load model results
    with open('detailed_nested_cv_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Load patch embeddings
    with open('response_features_portal_tract.pkl', 'rb') as f:
        response_features = pickle.load(f)
    
    with open('no_response_features_portal_tract.pkl', 'rb') as f:
        no_response_features = pickle.load(f)
    
    log_output(f"Loaded {len(response_features['embeddings'])} response patches and " 
              f"{len(no_response_features['embeddings'])} no-response patches")
    
    return results, response_features, no_response_features

def find_best_model(results):
    """Find the best performing model from results"""
    raw_results = results['raw_results']
    best_model_type = None
    best_model_name = None
    best_balanced_acc = -1
    
    for model_type in raw_results:
        for model_name, metrics in raw_results[model_type].items():
            if not metrics['balanced_accuracy']:
                continue
            
            mean_balanced_acc = np.mean(metrics['balanced_accuracy'])
            
            if mean_balanced_acc > best_balanced_acc:
                best_balanced_acc = mean_balanced_acc
                best_model_type = model_type
                best_model_name = model_name
    
    log_output(f"Best model: {best_model_type} {best_model_name} with balanced accuracy {best_balanced_acc:.4f}")
    
    # Get the most common hyperparameters across folds
    best_params = raw_results[best_model_type][best_model_name]['best_params']
    
    # Count parameter frequencies
    param_counts = {}
    for params_dict in best_params:
        for param, value in params_dict.items():
            if param not in param_counts:
                param_counts[param] = {}
            
            if value not in param_counts[param]:
                param_counts[param][value] = 0
            
            param_counts[param][value] += 1
    
    # Get most common value for each parameter
    most_common_params = {}
    for param, counts in param_counts.items():
        most_common_value = max(counts.items(), key=lambda x: x[1])[0]
        most_common_params[param] = most_common_value
    
    return best_model_type, best_model_name, most_common_params

def create_model(model_name, params):
    """Create a model with the specified parameters"""
    if model_name == 'lr':
        model = LogisticRegression(**params, max_iter=1000)
    elif model_name == 'svm':
        model = SVC(probability=True, **params)
    elif model_name == 'rf':
        model = RandomForestClassifier(**params)
    elif model_name == 'gb':
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Wrap in pipeline with scaling for lr and svm
    if model_name in ['lr', 'svm']:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    
    return model

def score_patches(model, response_features, no_response_features, use_pathology_only=False):
    """Score all patches using the trained model"""
    log_output("Scoring all patches...")
    
    # Extract embeddings
    response_embeddings = response_features['embeddings']
    no_response_embeddings = no_response_features['embeddings']
    
    # Combine all embeddings
    all_embeddings = np.vstack([response_embeddings, no_response_embeddings])
    
    # Score with the model
    if hasattr(model, 'predict_proba'):
        all_scores = model.predict_proba(all_embeddings)
        # Get probability for class 1 (No Response)
        all_scores_class1 = all_scores[:, 1]
        # Class 0 (Response) scores are 1 - class1 scores
        all_scores_class0 = 1 - all_scores_class1
    else:
        # For models that don't provide probabilities, use decision function if available
        if hasattr(model, 'decision_function'):
            all_scores_class1 = model.decision_function(all_embeddings)
            all_scores_class0 = -all_scores_class1  # Invert for class 0
        else:
            # Fallback to binary predictions
            all_preds = model.predict(all_embeddings)
            all_scores_class1 = all_preds.astype(float)
            all_scores_class0 = 1 - all_preds
    
    # Split scores back for response and no-response
    response_scores_class0 = all_scores_class0[:len(response_embeddings)]
    response_scores_class1 = all_scores_class1[:len(response_embeddings)]
    
    no_response_scores_class0 = all_scores_class0[len(response_embeddings):]
    no_response_scores_class1 = all_scores_class1[len(response_embeddings):]
    
    return {
        'response': {
            'class0_scores': response_scores_class0,
            'class1_scores': response_scores_class1
        },
        'no_response': {
            'class0_scores': no_response_scores_class0,
            'class1_scores': no_response_scores_class1
        }
    }

def get_top_patches(scores, response_features, no_response_features, top_n=10):
    """Get top N patches most likely for each class"""
    log_output(f"Getting top {top_n} patches for each class...")
    
    # ---------- For Class 0 (Response) ----------
    # From Response group
    indices_resp_class0 = np.argsort(scores['response']['class0_scores'])[::-1][:top_n]
    top_resp_class0 = {
        'paths': [response_features['paths'][i] for i in indices_resp_class0],
        'scores': scores['response']['class0_scores'][indices_resp_class0],
        'source': ['response'] * top_n
    }
    
    # From No Response group
    indices_noresp_class0 = np.argsort(scores['no_response']['class0_scores'])[::-1][:top_n]
    top_noresp_class0 = {
        'paths': [no_response_features['paths'][i] for i in indices_noresp_class0],
        'scores': scores['no_response']['class0_scores'][indices_noresp_class0],
        'source': ['no_response'] * top_n
    }
    
    # Combine and take overall top N for class 0
    combined_paths_class0 = top_resp_class0['paths'] + top_noresp_class0['paths']
    combined_scores_class0 = np.concatenate([top_resp_class0['scores'], top_noresp_class0['scores']])
    combined_source_class0 = top_resp_class0['source'] + top_noresp_class0['source']
    
    indices_class0 = np.argsort(combined_scores_class0)[::-1][:top_n]
    top_class0 = {
        'paths': [combined_paths_class0[i] for i in indices_class0],
        'scores': combined_scores_class0[indices_class0],
        'source': [combined_source_class0[i] for i in indices_class0]
    }
    
    # ---------- For Class 1 (No Response) ----------
    # From Response group
    indices_resp_class1 = np.argsort(scores['response']['class1_scores'])[::-1][:top_n]
    top_resp_class1 = {
        'paths': [response_features['paths'][i] for i in indices_resp_class1],
        'scores': scores['response']['class1_scores'][indices_resp_class1],
        'source': ['response'] * top_n
    }
    
    # From No Response group
    indices_noresp_class1 = np.argsort(scores['no_response']['class1_scores'])[::-1][:top_n]
    top_noresp_class1 = {
        'paths': [no_response_features['paths'][i] for i in indices_noresp_class1],
        'scores': scores['no_response']['class1_scores'][indices_noresp_class1],
        'source': ['no_response'] * top_n
    }
    
    # Combine and take overall top N for class 1
    combined_paths_class1 = top_resp_class1['paths'] + top_noresp_class1['paths']
    combined_scores_class1 = np.concatenate([top_resp_class1['scores'], top_noresp_class1['scores']])
    combined_source_class1 = top_resp_class1['source'] + top_noresp_class1['source']
    
    indices_class1 = np.argsort(combined_scores_class1)[::-1][:top_n]
    top_class1 = {
        'paths': [combined_paths_class1[i] for i in indices_class1],
        'scores': combined_scores_class1[indices_class1],
        'source': [combined_source_class1[i] for i in indices_class1]
    }
    
    return top_class0, top_class1

def extract_patient_id(path):
    """Extract patient ID from file path"""
    try:
        basename = os.path.basename(path)
        # Remove file extension
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('-')
        
        if len(parts) >= 2:
            # More robust patient ID extraction
            patient_id = parts[1].strip()
            if patient_id and patient_id.replace('_', '').isalnum():
                return patient_id
        
        return "unknown"
    except Exception as e:
        return "unknown"

def visualize_top_patches(top_class0, top_class1, output_dir="top_patches"):
    """Visualize and save top patches for each class"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "class0_response"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "class1_no_response"), exist_ok=True)
    
    # Function to create a figure with patches
    def create_patch_figure(top_patches, title, filename):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, (path, score, source) in enumerate(zip(top_patches['paths'], 
                                                    top_patches['scores'], 
                                                    top_patches['source'])):
            if i >= 10:
                break
                
            patient_id = extract_patient_id(path)
            img = Image.open(path)
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(f"Score: {score:.3f}\nSource: {source}\nPatient: {patient_id}")
            axes[i].axis('off')
            
            # Copy the file to output directory
            img_filename = f"{i+1:02d}_score_{score:.3f}_{os.path.basename(path)}"
            if title.startswith("Class 0"):
                save_path = os.path.join(output_dir, "class0_response", img_filename)
            else:
                save_path = os.path.join(output_dir, "class1_no_response", img_filename)
            shutil.copy2(path, save_path)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create and save figures
    create_patch_figure(top_class0, "Class 0 (Response) - Top 10 Patches", "top_response_patches.png")
    create_patch_figure(top_class1, "Class 1 (No Response) - Top 10 Patches", "top_no_response_patches.png")
    
    # Save metadata
    class0_df = pd.DataFrame({
        'rank': range(1, len(top_class0['paths']) + 1),
        'path': top_class0['paths'],
        'score': top_class0['scores'],
        'source': top_class0['source'],
        'patient_id': [extract_patient_id(p) for p in top_class0['paths']]
    })
    
    class1_df = pd.DataFrame({
        'rank': range(1, len(top_class1['paths']) + 1),
        'path': top_class1['paths'],
        'score': top_class1['scores'],
        'source': top_class1['source'],
        'patient_id': [extract_patient_id(p) for p in top_class1['paths']]
    })
    
    class0_df.to_csv(os.path.join(output_dir, "class0_response_top_patches.csv"), index=False)
    class1_df.to_csv(os.path.join(output_dir, "class1_no_response_top_patches.csv"), index=False)
    
    log_output(f"Visualization saved to {output_dir}")
    log_output(f"Top 10 patches for class 0 (Response):")
    for i, (score, source) in enumerate(zip(top_class0['scores'], top_class0['source'])):
        log_output(f"  {i+1}. Score: {score:.4f}, Source: {source}")
    
    log_output(f"Top 10 patches for class 1 (No Response):")
    for i, (score, source) in enumerate(zip(top_class1['scores'], top_class1['source'])):
        log_output(f"  {i+1}. Score: {score:.4f}, Source: {source}")

def train_best_pathology_model(results, response_features, no_response_features):
    """Train the best pathology-only model to score patches"""
    # Find the best pathology model
    best_model_type, best_model_name, best_params = find_best_model({'raw_results': 
                                                                   {'Pathology': results['raw_results']['Pathology']}})
    
    # Create the model
    model = create_model(best_model_name, best_params)
    
    # Prepare training data
    X_response = response_features['embeddings']
    y_response = response_features['labels']
    
    X_no_response = no_response_features['embeddings']
    y_no_response = no_response_features['labels']
    
    X_all = np.vstack([X_response, X_no_response])
    y_all = np.concatenate([y_response, y_no_response])
    
    # Train the model
    log_output(f"Training {best_model_name} model with parameters: {best_params}")
    model.fit(X_all, y_all)
    
    return model

def main():
    log_output("Starting top patch identification...")
    
    # Load data
    results, response_features, no_response_features = load_data()
    
    # Train a pathology-specific model for patch scoring
    model = train_best_pathology_model(results, response_features, no_response_features)
    
    # Score all patches
    patch_scores = score_patches(model, response_features, no_response_features)
    
    # Get top patches for each class
    top_class0, top_class1 = get_top_patches(patch_scores, response_features, no_response_features, top_n=10)
    
    # Visualize and save results
    visualize_top_patches(top_class0, top_class1)
    
    log_output("Process completed successfully!")

if __name__ == "__main__":
    main()