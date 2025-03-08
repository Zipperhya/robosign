import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
from sklearn.preprocessing import label_binarize
class Metrics:
    def __init__(self, pred, target, num_class, save_path=None, exp_name=None):

        '''
        The metrics can be used to calculate confusion matrix, F1 score, and AUC.
        Input:
            pred: torch.Tensor, the predicted labels
            target: torch.Tensor, the ground truth labels
            num_class: int, the number of classes
            save_path: str, the path to save the confusion matrix and ROC curve pictures
        '''

        # self.pred = pred.cpu().numpy()
        # self.target = target.cpu().numpy()
        self.pred = pred
        self.target = target
        self.num_class = num_class
        self.save_path = save_path
        self.exp_name = exp_name


    def confusion_matrix(self):
        '''
        Calculate the confusion matrix and plot the confusion matrix
        '''

        conf_matrix = confusion_matrix(self.target, self.pred)
        conf_matrix_prob = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
        if self.save_path is not None:
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix_prob, annot=True, fmt='.2f', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(self.save_path, self.exp_name + 'confusion_matrix.png'))
            plt.close()

        return conf_matrix_prob

    def F1_score(self):
        '''
        Calculate the F1 score
        '''
        f1 = f1_score(self.target, self.pred, average='weighted')
        return f1

    def AUC(self):
        '''
        Calculate the AUC and plot the ROC curve
        '''
        classes = np.arange(self.num_class)
        target = label_binarize(self.target, classes=classes)
        pred = label_binarize(self.pred, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(target.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(target[:, i], pred[:, i])
            roc_auc[i] = roc_auc_score(target[:, i], pred[:, i])

        fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), pred.ravel())
        roc_auc["micro"] = roc_auc_score(target, pred, average="micro")

        if self.save_path is not None:
            plt.figure()
            for i in range(target.shape[1]):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_path, 'roc_curve.png'))

        return fpr, tpr, roc_auc


