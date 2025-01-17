import torch
import torch.nn as nn
from torchvision.models import vgg19


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.vgg19_model = vgg19(pretrained=True)  # pre trained vgg19 load
        self.vgg19_features = self.vgg19_model.features

        # TODO: 추출해야 하는 layer 지정
        content_layers = []
        style_laeyrs = []

    def forward(self, x: torch.Tensor, status: str):
        outputs = []

        if status == "content":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                # TODO: 지정된 번호의 layer를 통과하면 outputs list에 추가
        elif status == "style":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                # TODO: 지정된 번호의 layer를 통과하면 outputs list에 추가

        return outputs
