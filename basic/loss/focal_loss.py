#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/8 下午6:42
# @Author : PH
# @Version：V 0.1
# @File : focal_loss.py
# @desc :
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, use_sigmoid=True, reduction='mean', **kwargs):
        """FL = alpha_weight * focal_weight * ce_loss = −αt(1 − pt)^γ *log(pt)"""
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid

    def py_sigmoid_focal_loss(self,
                              pred,
                              target,
                              weights=None,
                              avg_factor=None
                              ):
        pred = pred.squeeze()
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
        ) * focal_weight
        focal_loss = weights * loss
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.sum() / avg_factor
        elif self.reduction == "sum":
            return focal_loss.sum()
        return loss

    def forward(self, inputs, targets, avg_factor, weights=None):
        """

        Args:
            inputs: B,N,C or B,C
            targets: B,N or B. The value of targets should in range [0, C-1]
            weights: B*N, or B, . weights for each inputs,default to alpha weight
        Returns:

        """
        if self.use_sigmoid:
            loss = self.py_sigmoid_focal_loss(pred=inputs,
                                              target=targets,
                                              weights=weights,
                                              avg_factor=avg_factor
                                              )
            return loss
        else:
            device = inputs.device
            if type(targets) != torch.long:
                targets = targets.to(dtype=torch.long)
            C = inputs.size(-1)
            alpha_weight_base = self._alpha_weight(num_class=C).to(device)  # [alpha, 1 - alpha, 1 - alpha,....,1-alpha]
            preds = inputs.view(-1, C)  # B*N, C
            logp = F.log_softmax(preds, dim=-1)
            p = torch.exp(logp)

            # lop, p and weight for each sample
            logpt = torch.gather(logp, dim=1, index=targets.view(-1, 1))  # B*N, 1
            pt = torch.gather(p, dim=1, index=targets.view(-1, 1))
            alpha_weight = torch.gather(alpha_weight_base, dim=0, index=targets.view(-1))  # B*N,
            # if weights is None:
            #     weights = torch.gather(alpha_weight, dim=0, index=targets.view(-1))  # B*N,
            # else:
            #     if not isinstance(weights, torch.Tensor):
            #         weights = torch.tensor(weights, device=device, dtype=torch.float)

            # focal loss
            focal_loss = -torch.pow(1 - pt, self.gamma) * logpt
            focal_loss = weights.view(-1) * alpha_weight * focal_loss.squeeze()
            if self.reduction == "none":
                return focal_loss
            elif self.reduction == "mean":
                return focal_loss.sum() / avg_factor
            elif self.reduction == "sum":
                return focal_loss.sum()

    def _alpha_weight(self, num_class):
        weight = torch.ones(num_class, dtype=torch.float)
        weight[0] = self.alpha
        weight[1:] -= self.alpha
        return weight


if __name__ == '__main__':
    # test
    inputs = torch.randn(8, 100, 4, requires_grad=True)
    targets = torch.randint(0, 3, size=(8, 100), dtype=torch.long)
    fl = FocalLoss(gamma=2, alpha=0.25)
    ce = nn.CrossEntropyLoss()
    print("Focal loss:", fl(inputs, targets))
    print("Cross entropy loss:", ce(inputs.permute(0, 2, 1), targets))
