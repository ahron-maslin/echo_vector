"""
Evaluation metrics for comparing audio vectors and embeddings.
"""
import numpy as np
import numpy.typing as npt

def cosine_similarity(
    vec1: npt.NDArray[np.float32], 
    vec2: npt.NDArray[np.float32]
) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")
        
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def euclidean_distance(
    vec1: npt.NDArray[np.float32], 
    vec2: npt.NDArray[np.float32]
) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Euclidean distance.
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")
        
    distance = np.linalg.norm(vec1.flatten() - vec2.flatten())
    return float(distance)
