import torch

class exponential_mse():
    def __init__(self, gamma=2.0, reduction='mean'):
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, y_true, y_pred):
        error = torch.abs(y_true - y_pred)  # Absolute error
        loss = (error ** self.gamma)  # Exponential loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class saturated_l1():
    def __init__(self, threshold=0.1, below=False):
        self.threshold = threshold
        self.below = below

    def __call__(self, y_true, y_pred):
        loss = torch.abs(y_true - y_pred)
        if self.below:
            saturated_loss = torch.where(loss < self.threshold, self.threshold, loss)
        else:
            saturated_loss = torch.where(loss > self.threshold, self.threshold, loss)
        return torch.mean(saturated_loss)

class saturated_l2():
    def __init__(self, threshold=0.1, below=False):
        self.threshold = threshold
        self.below = below

    def __call__(self, y_true, y_pred):
        loss = (y_true - y_pred)**2
        if self.below:
            saturated_loss = torch.where(loss < self.threshold, self.threshold, loss)
        else:
            saturated_loss = torch.where(loss > self.threshold, self.threshold, loss)
        return torch.mean(saturated_loss)

