import torch
import torch.nn as nn
from torchvision.models import vgg19


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        # TODO: pre-trained VGG19 load
        # TODO: 추출해야 하는 layer 지정
        pass

    def forward(self):
        # TODO: 지정된 layer를 통과한 feature map 추출
        # TODO: 모델 출력: (content, style)feature map을 담고 있는 list 형태
        pass
