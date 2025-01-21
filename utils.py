import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image


VGG19_MEAN = [0.485, 0.456, 0.406]
VGG19_STD = [0.229, 0.224, 0.225]


def pre_processing(image: torch.Tensor) -> torch.Tensor:
    """
    이미지를 모델의 입력 형태에 맞게 변환하는 함수

    Args:
        iamge (torch.Tensor): 모델의 입력하려는 이미지 (ch, h, w)

    Returns:
        torch.Tensor: (b, ch, h, w) 형태의 이미지
    """
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=VGG19_MEAN, std=VGG19_STD),
        ]
    )
    image_tensor: torch.Tensor = transform(image)
    return image_tensor.unsqueeze(0)


def post_processing():
    # TODO: tensor device to cpu + tensor2numpy
    # TODO: (b, ch, h, w) -> (ch, h, w)
    # TODO: (ch, h, w) -> (h, w, ch) for PIL format
    # TODO: de normalize
    # TODO: dtype: float -> unit8
    # TODO: numpy -> PIL image
    pass
