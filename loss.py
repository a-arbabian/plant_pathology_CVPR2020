# @Author: yican, yelanlan
import torch.nn as nn
import torch


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        loss = torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))
        return loss
