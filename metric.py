import scanpy as sc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from ucegen.helpers.model_utils import ClassifierMLP

def fit_with_progress(model, X_train, y_train, n_estimators):
    model.set_params(warm_start=True)  # Enable warm start to fit trees incrementally
    with tqdm(total=n_estimators, desc="Training Random Forest") as pbar:
        for i in range(1, n_estimators + 1):
            model.set_params(n_estimators=i)  # Incrementally add trees
            model.fit(X_train, y_train)  # Fit the model with the added tree
            pbar.update(1)  # Update the progress bar for each tree
    return model

def discriminator_score(pred, true, num_trees=100):
    """
    Train a random forest classifier as a discriminator binary model between original and generated embeddings.
    Args:
        pred: The generated embeddings.
        true: The original embeddings.
    Returns:
        The AUROC score.
    """
    X = np.concatenate([true, pred], axis=0)
    y = np.concatenate([np.zeros(len(true)), np.ones(len(pred))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    rfc1 = RandomForestClassifier(n_estimators=num_trees, 
                                  max_depth=5,      
                                  random_state=42)
    
    rfc1 = fit_with_progress(rfc1, X_train, y_train, n_estimators=num_trees)
    y_probs = rfc1.predict_proba(X_test)[:, 1]     
    return roc_auc_score(y_test, y_probs), rfc1

def get_cos_sim(x, y):
    """
    Direct calculation of cosine similarity between two vectors.
    Helper function for mean_cosine_score.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def mean_cosine_score(pred, true):
    """
    Calculate the cosine similarity between two groups of samples.
    Both args are of shape (n_samples, n_features).
    """
    centroid_Y = np.mean(true, axis=0, keepdims=True)
    cos_sim_scores = []
    for i in range(pred.shape[0]):
        x_i = pred[i]
        y = centroid_Y.flatten()
        cos_sim_calc = get_cos_sim(x_i, y)
        cos_sim_scores.append(cos_sim_calc)
    
    return np.mean(cos_sim_scores)

def compute_mse(pred, true):
    """Computes mean squared error between x and y."""
    return mean_squared_error(pred, true)

def compute_wasserstein(pred, true, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    # Compute cost
    geom_pred_true = pointcloud.PointCloud(pred, true, epsilon=epsilon)
    ot_prob = linear_problem.LinearProblem(geom_pred_true)

    # Solve ot problem
    solver = sinkhorn.Sinkhorn()
    out_pred_true = solver(ot_prob)

    # Return regularized ot cost
    return out_pred_true.reg_ot_cost

def get_name_to_function_mapping():
    return {
        "mse": compute_mse,
        "real_gen_disc": discriminator_score,
        "cosine": mean_cosine_score,
        "wasserstein": compute_wasserstein,
    }
