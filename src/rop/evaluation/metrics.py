from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict[str, Any]:
    """Computes standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'report_dict': report,
        'report_str': report_str,
        'confusion_matrix': cm.tolist()
    }
