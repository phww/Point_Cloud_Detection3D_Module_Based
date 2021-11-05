#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/8 下午7:04
# @Author : PH
# @Version：V 0.1
# @File : smoothL1_loss.py
# @desc :
import torch
import torch.nn as nn


def smooth_l1_loss(diff, beta):
    # if beat < 1e-5.use L1 loss
    if beta < 1e-5:
        loss = torch.abs(diff)
    else:
        n = torch.abs(diff)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    return loss


class SmoothL1Loss(nn.Module):
    """
    Pytroch also have implement of torch.nn.SmoothL1Loss().See
    https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=l1#torch.nn.SmoothL1Loss
    """

    def __init__(self, beta, reduction='mean', **kwargs):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        # self.code_weight = torch.tensor(code_weights, dtype=torch.float) if code_weights is not None else None

    def forward(self, inputs, targets, weights=None, avg_factor=128):
        """

        Args:
            inputs: B, N, 7 or N, 7
            targets: B, N, 7 or N, 7
            code_weights: weights for code dim

        Returns:

        """
        assert inputs.size() == targets.size()
        diff = inputs - targets
        loss = smooth_l1_loss(diff, self.beta)
        # if self.code_weights is not None:
        #     assert self.code_weights.shape[0] == loss.shape[-1]
        #     loss = loss * self.code_weights.unsqueeze(-1)
        if weights is not None:
            assert weights.shape == loss.shape[:2]
            weights = weights.unsqueeze(dim=2).repeat(1, 1, 7)
            loss = weights * loss
        if self.reduction == 'mean':
            return loss.sum() / avg_factor
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()


if __name__ == '__main__':
    preds = torch.randn(100, 7, requires_grad=True)
    targets = torch.randn(100, 7)
    loss_fn = SmoothL1Loss(beta=0.5)
    loss = loss_fn(preds, targets)
    print(f"loss:", loss)
    print(f"grad:", loss.backward())
    # print(loss.grad)
    torch_loss = torch.nn.SmoothL1Loss(beta=0.5)
    print(f"torch loss:", torch_loss(preds, targets))
    print(f"grad:", torch_loss(preds, targets).backward())
