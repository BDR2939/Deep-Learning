import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1
        N = x_scores.size(0)
        margin = x_scores - x_scores[torch.arange(N), y].view(N, 1) + self.delta
        margin[torch.arange(N), y] = 0
        loss_per_sample = torch.nn.functional.relu(margin)
        loss = loss_per_sample.sum() / N

        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['x_scores'] = x_scores
        self.grad_ctx['margin'] = margin

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).
        """
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        x_scores = self.grad_ctx['x_scores']
        margin = self.grad_ctx['margin']
        G = torch.zeros(x_scores.shape)
        G[margin > 0] = 1
        G[torch.arange(x.shape[0]), y] = -torch.sum(G, dim=1)
        return torch.mm(x.t(), G) / x.shape[0]
