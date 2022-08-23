import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


def ssl1train_loss(pred, target, alpha):
    clip_max = 1.0
    clip_min = 1 / 1024.

    sigma_gt = torch.abs(pred['mu'] - target)
    z = torch.randn(target.shape).to(target.device) # Gaussian random variables (4D tensor)
    loss_main = torch.mean(torch.clamp(torch.abs(pred['mu']+sigma_gt*z.detach()-target), clip_min, clip_max)) #
    loss_aux = torch.mean(torch.clamp(torch.abs(pred['sigma'] - target), clip_min, clip_max))
    
    return loss_main + alpha * loss_aux

def ssl1test_loss(pred, target):
    clip_max = 1.0
    clip_min = 1 / 1024.

    sigma_gt = torch.abs(pred['mu'] - target)
    loss_main = torch.mean(torch.clamp(torch.abs(pred['mu']-target), clip_min, clip_max)) 
    loss_aux = torch.mean(torch.clamp(torch.abs(pred['sigma'] - sigma_gt), clip_min, clip_max))
    
    return loss_main + loss_aux

class SSL1LossTrain(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSL1LossTrain, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * ssl1train_loss(
            pred, target, weight, reduction=self.reduction)


class SSL1LossTest(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSL1LossTest, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return ssl1test_loss(pred, target)
'''
@weighted_loss
def l1_loss(pred, target):
    clip_max = 1.0
    clip_min = 1 / 1024.

    return torch.mean(torch.clamp(torch.abs(pred - target), clip_min, clip_max))
    #return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
'''