import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train = torch.empty(0, dtype=torch.float32)
        y_train = torch.empty(0, dtype=torch.long)
        n_classes = 0
        for x, y in dl_train:
            x_train = torch.cat((x_train, x))
            y_train = torch.cat((y_train, y))
            n_classes = max(n_classes, y.max()) + 1

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            nearest_neighbors = torch.topk(dist_matrix[:, i], k=self.k, largest=False).indices
            neighbor_labels = self.y_train[nearest_neighbors]
            unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
            most_common_label = unique_labels[counts.argmax()]
            y_pred[i] = most_common_label
        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    N1, D = x1.size()
    N2, D = x2.size()
    x1 = x1.view(N1, 1, D)
    x2 = x2.view(1, N2, D)
    dists = ((x1 - x2) ** 2).sum(dim=2).sqrt()
    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    correct = (y == y_pred).sum()
    return correct / len(y)


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        k_accuracies = []
        num_samples = len(ds_train)
        fold_size = num_samples // num_folds
        indices = torch.randperm(num_samples)
        for fold in range(num_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size
            valid_idx = indices[start_idx:end_idx]
            train_idx = torch.cat((indices[:start_idx], indices[end_idx:]))
            dl_train = DataLoader(ds_train, sampler=SubsetRandomSampler(valid_idx))
            model.train(dl_train)
            dl_valid = DataLoader(ds_train, sampler=SubsetRandomSampler(train_idx))
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            y_pred = model.predict(x_valid)
            k_accuracies.append(accuracy(y_valid, y_pred))
        accuracies.append(k_accuracies)

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
