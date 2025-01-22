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

        # -- trained object setting -> 학습되는 대상을 지정해서 학습을 진행하도록 업데이트 예정
        output = torch.randn(1, 3, 512, 512).to(device)
        output.requires_grad_(True)

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
