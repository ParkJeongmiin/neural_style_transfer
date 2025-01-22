import torch
import torch.optim as optim

from PIL import Image

from model import StyleTransfer
from loss import ContentLoss, StyleLoss
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

        # -- model, loss, optimizer setting
        model = StyleTransfer().eval().to(device)
        content_loss = ContentLoss()
        style_loss = StyleLoss(style_feature_maps_num=model.style_feature_maps_num)
        optimizer = optim.Adam(output, lr=args.lr)
        # hyperparameter 설정

        # trainloop
        ## model apply
        ## loss calculate
        ## optimizer update
        ## loss log

        ## data post processing : tensor to image
        ## save generated image
        pass
