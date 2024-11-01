from sklearn.metrics import confusion_matrix, jaccard_score

def evaluate_model(y_test, predictions):
    """Evaluate the model and return metrics."""
    conf_matrix = confusion_matrix(y_test, predictions)
    jaccard = jaccard_score(y_test, predictions)
    
    TN = conf_matrix[0, 0]
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    FP = conf_matrix[0, 1]
    
    sp = TN / (TN + FP)  # True negative rate
    sn = TP / (TP + FN)  # Sensitivity

    return {
        "confusion_matrix": conf_matrix,
        "jaccard_score": jaccard,
        "true_negative_rate": sp,
        "sensitivity": sn
    }