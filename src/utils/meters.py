import datetime
from collections import defaultdict
from datetime import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    roc_curve, auc, RocCurveDisplay,
    explained_variance_score
)
import matplotlib.pyplot as plt

from src.data.utils import DX_DICT
from src.utils.misc import SmoothedValue
from src.utils.misc import forecasting_acc, MetricLogger


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def draw_roc_curve(true_labels, preds, n_classes, save_path=None):
    """
    Draw and save ROC curve using sklearn for multi-class classification.

    Args:
    - true_labels: The ground truth labels (numpy array).
    - preds: The predicted probabilities (numpy array).
    - n_classes: The number of classes in the classification task.
    - save_path: Optional file path to save the plot. If None, the plot will not be saved.
    """
    # Binarize the labels for multi-class ROC calculation
    true_labels_bin = label_binarize(true_labels, classes=list(range(n_classes)))

    # Initialize plot
    plt.figure()
    DX_DICT_REV = {v: k for k, v in DX_DICT.items()}
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        class_name = DX_DICT_REV[i]
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the class
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc,
            estimator_name=f'Class {class_name}'
        )
        display.plot(ax=plt.gca())  # Plot on the same axis

    plt.title('Receiver Operating Characteristic')

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    plt.show()
    return plt


class ClassTestMeter(MetricLogger):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
            self,
            ensemble_method="sum",
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ensemble_method = ensemble_method
        self.topk_accs = []
        self.stats = {}
        self.subject_preds, self.subject_targets = dict(), dict()
        self.subject_count = dict()

    def store_predictions(self, preds, labels, subject_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            subject_ids (tensor): subject indices of the current batch, dimension is
                N.
        """
        for i, id in enumerate(subject_ids):
            if id in self.subject_preds:
                assert id in self.subject_targets, f'RID {id} not in targets'
                assert torch.equal(
                    self.subject_targets[id].type(torch.FloatTensor),
                    labels[i].type(torch.FloatTensor),
                )
                if self.ensemble_method == 'sum':
                    self.subject_preds[id] += preds[i]
                elif self.ensemble_method == 'max':
                    self.subject_preds[id] = torch.maximum(self.subject_preds[id], preds[i])
                else:
                    raise NotImplementedError(f'Ensemble Method {self.ensemble_method} is not supported.')
                self.subject_count[id] += 1
            else:
                self.subject_preds[id] = preds[i]
                self.subject_targets[id] = labels[i]
                self.subject_count[id] = 1

    def finalize_metrics(self, target_shape=None, ks=(1, 2), args=None):
        """
        Calculate and log the final ensembled metrics including top-k accuracy,
        precision, recall, and ROC AUC score.

        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) corresponds to top-1 and top-5 accuracy.
        """
        assert self.subject_preds.keys() == self.subject_targets.keys()

        # Store stats
        self.stats = {"split": "test_final"}

        # Convert predictions and targets to appropriate formats
        preds_list = [arr for key, arr in self.subject_preds.items()]
        targets_list = [arr for key, arr in self.subject_targets.items()]

        preds = F.softmax(torch.FloatTensor(np.array(preds_list)), dim=1)
        targets = torch.FloatTensor(np.array(targets_list))
        preds_np = preds.cpu().numpy()
        # Top-k Accuracy
        num_topks_correct = topks_correct(preds, targets, ks)
        topks = [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            self.stats[f'acc{k}'] = "{:.{prec}f}".format(topk, prec=2)

        # Precision, Recall, ROC AUC (for binary/multi-class)
        pred_labels = preds.argmax(dim=1).cpu().numpy()  # Convert to predicted class labels
        true_labels = targets.cpu().numpy()

        # Calculate precision and recall
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')

        # Calculate ROC AUC for multi-class classification (One-vs-Rest)
        roc_auc = roc_auc_score(true_labels, preds_np, multi_class='ovo', average='macro')

        # Store precision, recall, and roc_auc in the stats
        self.stats['precision'] = "{:.2f}".format(precision * 100)
        self.stats['recall'] = "{:.2f}".format(recall * 100)
        self.stats['roc_auc'] = "{:.2f}".format(roc_auc * 100)
        print(self.stats)

        # Draw and save ROC curve
        n_classes = preds.shape[1]
        roc_curve_save_path = None
        if args:
            roc_curve_save_path = os.path.join('/'.join(args.finetune.split('/')[:-1]), f'roc_curve.png')
        draw_roc_curve(true_labels, preds_np, n_classes, save_path=roc_curve_save_path)
        return self.stats


class PredTestMeter(MetricLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = None
        self.targets = None
        self.target_shape = None

    def store_predictions(self, pred, target):
        """
        used to calculate overall metrics (as a single batch), instead of averaging
        :param pred:
        :param target:
        :return:
        """
        if self.preds is not None and self.targets is not None:
            self.preds = torch.cat(
                tensors=[self.preds, pred],
                dim=0
            )
            self.targets = torch.cat(
                tensors=[self.targets, target],
                dim=0
            )
        else:
            self.preds, self.targets = pred, target

    def finalize_metrics(self, target_shape=None):
        # TODO: early prediction
        if target_shape:
            target_shape = tuple([self.preds.shape[0]] + list(target_shape[1:]))
        metrics = forecasting_acc(
            self.preds,
            self.targets,
            target_shape
        )
        if target_shape is None:
            # Add explained variance ratio
            preds_np = self.preds.cpu().numpy().reshape(-1, self.preds.shape[-1])
            targets_np = self.targets.cpu().numpy().reshape(-1, self.targets.shape[-1])

            explained_var = explained_variance_score(targets_np, preds_np, multioutput='uniform_average')

            # Add explained variance ratio to metrics
            metrics['explained_variance_ratio'] = explained_var
        self.preds = None
        self.targets = None
        return metrics
