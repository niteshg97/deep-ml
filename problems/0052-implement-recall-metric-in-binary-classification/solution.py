import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # recall = Tp/(tp+fn)
    # Your code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1) )
    fn = np.sum ((y_true == 1 )& (y_pred == 0))
    if (tp+fn) == 0:
        return 0
    else:
        ans = tp / (tp+fn)
        return ans 
    
