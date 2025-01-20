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
    def __init__(
        self,
        current_style_feature_maps,
        target_style_feature_maps,
        style_feature_maps_num,
        gram_normalize=True,
    ):
        super(StyleLoss, self).__init__()
        self.gram_normalize = gram_normalize
        self.current_style_feature_maps = current_style_feature_maps
        self.target_style_feature_maps = target_style_feature_maps
        self.style_feature_maps_num = style_feature_maps_num

    def gram_matrix(self, input: torch.Tensor):
        (b, ch, h, w) = input.size()
        features = input.view(b, ch, h * w)
        features_T = features.transpose(1, 2)
        gram = features.bmm(features_T)
        if self.gram_normalize:
            gram /= ch * h * w
        return gram

    def forward(self):
        current_style_representation = [
            self.gram_matrix(x) for x in self.current_style_feature_maps
        ]

        target_style_representation = [
            self.gram_matrix(x) for x in self.target_style_feature_maps
        ]

        style_loss = 0.0
        for gram_x, gram_y in zip(
            current_style_representation, target_style_representation
        ):
            style_loss += torch.nn.MSELoss(reduction="sum")(gram_x, gram_y)
        style_loss /= len(self.style_feature_maps_num)
        return style_loss
