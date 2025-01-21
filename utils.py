import torchvision.transforms as T

import numpy as np
from PIL import Image


VGG19_MEAN = [0.485, 0.456, 0.406]
VGG19_STD = [0.229, 0.224, 0.225]


def pre_processing(self, input):
    # TODO: image transform : resize, to tensor, normalize
    # TODO: (ch, h, w) -> (b, ch, h, w)
    pass


def post_processing():
    # TODO: tensor device to cpu + tensor2numpy
    # TODO: (b, ch, h, w) -> (ch, h, w)
    # TODO: (ch, h, w) -> (h, w, ch) for PIL format
    # TODO: de normalize
    # TODO: dtype: float -> unit8
    # TODO: numpy -> PIL image
    pass
