import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_data(data): 
    """
    Normalize the data to [-1, 1]. 
    """
    if isinstance(data, sc.AnnData):
        if 'X_uce' in data.obsm.keys():
            print("UCE embeddings found in obsm['X_uce']. Normalizing...")
            data_X = data.obsm['X_uce']
            min_val, max_val = np.min(data_X, axis=0), np.max(data_X, axis=0)
            data_norm = 2 * ((data_X - min_val) / (max_val - min_val)) - 1
            new_data = sc.AnnData(data_norm)
            new_data.obs = data.obs

        elif data.X.shape[1] == 1280:
            print("UCE embeddings found in X. Normalizing...")
            data_X = data.X
            min_val, max_val = np.min(data_X, axis=0), np.max(data_X, axis=0)
            data.X = 2 * ((data_X - min_val) / (max_val - min_val)) - 1
            new_data = data
            
        else:
            raise ValueError("Data does not have UCE embeddings. Please pass in an adata object with UCE embeddings.")
        
    else:
        min_val, max_val = np.min(data, axis=0), np.max(data, axis=0)
        data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        new_data = data.copy()

    return new_data, min_val, max_val

def unnormalize_data(array, min_val, max_val):
    array = ((array + 1) / 2) * (max_val - min_val) + min_val
    return array

def subsample_adata(adata, column='coarse_cell_type', n_samples=10000):
    """
    Subsample the AnnData object to have equal number of samples for each class.
    """
    adata = adata[adata.obs[column] != 'Missing' or adata.obs[column] != 'missing']
    unique_classes = adata.obs[column].unique()
    indices_to_keep = []

    for cls in unique_classes:
        cls_indices = adata.obs[adata.obs[column] == cls].index
        if len(cls_indices) > n_samples:
            cls_indices = np.random.choice(cls_indices, n_samples, replace=False)
        indices_to_keep.extend(cls_indices)
    
    return adata[indices_to_keep].copy()

def merge_adata(adata1, adata2):
    """
    Merge two AnnData objects.
    """
    try:
        adata = sc.concat([adata1, adata2])
        return adata
    except ValueError:
        adata1.var.index = adata1.var.index.astype(str)
        adata2.var.index = adata2.var.index.astype(str)

        adata1.var_names_make_unique()
        adata2.var_names_make_unique()

        shared_genes = set(adata1.var_names).intersection(set(adata2.var_names))
        shared_genes = list(shared_genes)

        adata1 = adata1[:, shared_genes]
        adata2 = adata2[:, shared_genes]

        return adata

def make_labels(adata, column):
    """
    Make labels for the adata object.
    Deterministic process since we are using LabelEncoder, which sorts.
    """
    return LabelEncoder().fit_transform(adata.obs[column].values).tolist()

def create_random_label_embeddings(labels, embedding_dim):
    """
    Create label embeddings for the labels.
    """
    unique_labels = np.unique(labels)
    label_to_embedding = {label: np.random.randn(embedding_dim).astype(np.float32) for label in unique_labels}
    return label_to_embedding

def create_averaged_label_embeddings(adata, encoded_labels, original_column_name, embedding_dim):
    """
    Create label embeddings for the labels by averaging the embeddings of the cells.
    """
    unique_encoded_labels = np.unique(encoded_labels)
    label_encoder = LabelEncoder()
    original_labels = label_encoder.fit_transform(adata.obs[original_column_name])
    
    label_to_embedding = {}
    for encoded_label in unique_encoded_labels:
        indices = np.where(original_labels == encoded_label)[0]
        label_to_embedding[encoded_label] = np.mean(adata.X[indices], axis=0)
    
    # Ensure all embeddings have the correct dimension
    for key in label_to_embedding:
        if label_to_embedding[key].shape[0] != embedding_dim:
            label_to_embedding[key] = label_to_embedding[key][:embedding_dim]
    
    return label_to_embedding
