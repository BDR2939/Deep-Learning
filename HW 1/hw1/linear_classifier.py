import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = torch.randn((n_features, n_classes), dtype=torch.float32) * weight_std

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, dim=1)
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        acc = (y == y_pred).sum().item() / len(y)
        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0
            total_samples = 0
            for x, y in dl_train:
                y_pred, x_scores = self.predict(x)
                loss = loss_fn.loss(x, y, x_scores, y_pred) + weight_decay * torch.sum(self.weights ** 2)
                grad = loss_fn.grad() + weight_decay * self.weights
                self.weights = self.weights - learn_rate * grad
                total_correct += (y == y_pred).sum().item()
                average_loss += loss.item() * len(x)
                total_samples += len(x)

            train_res.accuracy.append(total_correct / total_samples)
            train_res.loss.append(average_loss / total_samples)

            total_correct = 0
            average_loss = 0
            total_samples = 0
            for x, y in dl_valid:
                y_pred, x_scores = self.predict(x)
                loss = loss_fn.loss(x, y, x_scores, y_pred) + weight_decay * torch.sum(self.weights ** 2)
                total_correct += (y == y_pred).sum().item()
                average_loss += loss.item() * len(x)
                total_samples += len(x)

            valid_res.accuracy.append(total_correct / total_samples)
            valid_res.loss.append(average_loss / total_samples)
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        if has_bias:
            weights = self.weights[1:]
        else:
            weights = self.weights
        n_features, n_classes = weights.shape
        w_images = weights.reshape(n_classes, *img_shape)
        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.005, weight_decay=0.01)
    return hp
