import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns


cls_list = [
    "entailment", 
    "neutral", 
    "contradiction"
]
idx2cls = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def calc_stats(target_classes, calced_logits, dataset_type, logits):
    if logits:
        calced_probs = nn.Softmax(dim=-1)(torch.Tensor(calced_logits)).numpy()
    else:
        calced_probs = calced_logits
    calced_classes = calced_probs.argmax(axis=1).astype(int)

    precision_threshold = precision_score(target_classes, calced_classes, average="micro")
    recall_threshold = recall_score(target_classes, calced_classes, average="micro")
    f1_threshold = f1_score(target_classes, calced_classes, average="micro")
    balanced_accuracy_threshold = balanced_accuracy_score(target_classes, calced_classes)
    accuracy_threshold = accuracy_score(target_classes, calced_classes)

    precision_all_cls = precision_score(target_classes, calced_classes, average=None)
    recall_all_cls = recall_score(target_classes, calced_classes, average=None)
    f1_all_cls = f1_score(target_classes, calced_classes, average=None)

    cm = confusion_matrix(target_classes, calced_classes)
    
    metrics = {
        f"precision/{dataset_type}": precision_threshold, 
        f"recall/{dataset_type}": recall_threshold, 
        f"f1/{dataset_type}": f1_threshold, 
        f"balanced_accuracy/{dataset_type}": balanced_accuracy_threshold,
        f"accuracy/{dataset_type}": accuracy_threshold, 
    }

    for i, (p, r, f) in enumerate(zip(precision_all_cls, recall_all_cls, f1_all_cls)):
        metrics.update({f"precision_cls/{dataset_type}/{idx2cls[i]}": p})
        metrics.update({f"recall_cls/{dataset_type}/{idx2cls[i]}": r})
        metrics.update({f"f1_cls/{dataset_type}/{idx2cls[i]}": f})
    
    return metrics, cm


def plot_confusion_matrix(cm, epoch, classes=cls_list):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    fig = plt.gcf()
    return fig
