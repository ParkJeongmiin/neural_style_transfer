import torch
import torch.nn as nn
import torch.nn.functional as F


# Content loss
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        content_loss = F.mse_loss(input, target)
        return content_loss


# Style loss
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        pass

    def forward(self, input, target):
        pass
