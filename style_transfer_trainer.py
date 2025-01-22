import torch

from PIL import Image

from utils import pre_processing, post_processing


class StyleTransferTrainer:
    """
    _summary_
    """

    def __init__(self, content_dir, style_dir, args):
        self.content_dir = content_dir
        self.style_dir = style_dir

    def train(self, args):
        # -- device setting
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- data load
        content_image = Image.open(self.content_dir)
        content_image = pre_processing(content_image).to(device)

        style_image = Image.open(self.style_dir)
        style_image = pre_processing(style_image).to(device)

        # model(StyleTransfer) load

        # loss(ContentLoss, StyleLoss), optimizer load

        # hyperparameter 설정

        # trainloop
        ## model apply
        ## loss calculate
        ## optimizer update
        ## loss log

        ## data post processing : tensor to image
        ## save generated image
        pass
