import torch
import torch.nn as nn
from torchvision.models import vgg19


conv = {
    "conv1_1": 0,  # style
    "conv2_1": 5,  # style
    "conv3_1": 10,  # style
    "conv4_1": 19,  # style
    "conv5_1": 28,  # style
    "conv4_2": 21,  # content
}


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.vgg19_model = vgg19(pretrained=True)  # pre trained vgg19 load
        self.vgg19_features = self.vgg19_model.features

        # TODO: 추출해야 하는 layer 지정
        self.content_layers = [conv["conv4_2"]]
        self.style_layers = [
            conv["conv1_1"],
            conv["conv2_1"],
            conv["conv3_1"],
            conv["conv4_1"],
            conv["conv5_1"],
        ]
        self.style_feature_maps_num = len(self.style_layers)

    def forward(self, x: torch.Tensor, status: str):
        outputs = []

        if status == "content":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.content_layers:
                    outputs.append(x)
        elif status == "style":
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.style_layers:
                    outputs.append(x)

        return outputs
