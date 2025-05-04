from typing import Dict, Union
import numpy as np

def precision_recall(sketch, true_counts: Dict[Union[str, int], int], threshold: int):
    """Calculate precision and recall for heavy hitters"""
    true_hitters = {k for k, v in true_counts.items() if v >= threshold}
    detected = set()
    false_positives = 0
    
    for item in true_counts:
        est = sketch.estimate(str(item))  # Ensure string conversion
        if est >= threshold:
            detected.add(item)
            if item not in true_hitters:
                false_positives += 1
                
    tp = len(detected & true_hitters)
    precision = tp / (tp + false_positives) if (tp + false_positives) > 0 else 1.0
    recall = tp / len(true_hitters) if true_hitters else 1.0
    
    return precision, recall

def quantile_error(true_values: list, sketch, quantiles: list):
    """Calculate absolute errors for quantile estimation"""
    errors = []
    for q in quantiles:
        true_val = np.quantile(true_values, q)
        est_val = sketch.query(q)
        errors.append(abs(true_val - est_val))
    return errors