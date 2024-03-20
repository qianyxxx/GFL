# metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')