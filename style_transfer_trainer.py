import torch
import torch.optim as optim

from PIL import Image
from tqdm import tqdm

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

        # -- training
        for epoch in tqdm(range(args.epochs)):
            # -- data input to model
            content_features = model(output, "content")
            style_features = model(output, "style")

            target_content_features = model(content_image, "content")
            target_style_features = model(style_image, "style")

            # -- loss calculate
            loss_c, loss_s, loss_total = 0, 0, 0

            for content_representation, target_content_representation in zip(
                content_features, target_content_features
            ):
                loss_c += content_loss(
                    content_representation, target_content_representation
                )
            loss_s = style_loss(style_features, target_style_features)
            loss_total = args.alpha * loss_c + args.beta * loss_s

            # -- optimizer update
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # -- loss log by log_interval
            if epoch % args.log_interval == 0:
                print(
                    f"epoch: {epoch} | loss total: {loss_total} | content loss: {loss_c} | style loss: {loss_s}"
                )
                result = post_processing(output)
