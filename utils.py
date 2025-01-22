import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image


VGG19_MEAN = [0.485, 0.456, 0.406]
VGG19_STD = [0.229, 0.224, 0.225]


def pre_processing(image: torch.Tensor) -> torch.Tensor:
    """
    이미지를 모델의 입력 형태에 맞게 변환하는 함수입니다.

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


def post_processing(tensor: torch.Tensor) -> Image.Image:
    """
    모델의 출력을 사람이 확인할 수 있는 이미지 형태로 변환하는 함수입니다.

    Args:
        tensor (torch.Tensor): 모델의 출력 (b, ch, h, w)

    Returns:
        Image.Image: 모델의 결과 이미지 PIL (h, w, ch)
    """
    image = tensor.detach().to("cpu").numpy()
    image = image.squeeze(0)  # (b, ch, h, w) -> (ch, h, w)
    image = image.transpose(1, 2, 0)  # (ch, h, w) -> (h, w, ch)
    image = image * VGG19_STD + VGG19_MEAN
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)
