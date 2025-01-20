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
# TODO: Gram matrix
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        pass

    def gram_matrix(self, x: torch.Tensor):
        # TODO: Gram matrix 구현
        # input: (b, ch, h, w) -> (b, ch, h*w) -> (b, N, M)
        # input.T : (b, M, N)
        # input @ input.T : (b, N, N)
        # normalization(by official github): /= (ch * h * w)
        pass

    def forward(self, input, target):
        # TODO: input(gram matrix)와 target(gram matrix) MSE loss
        pass
