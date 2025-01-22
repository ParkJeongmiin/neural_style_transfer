import os
import argparse
import datetime
from importlib import import_module
from pytz import timezone


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data and model parameters directories
    # -- directory setting
    parser.add_argument(
        "--save_name", type=str, default="result", help="svave name(default: result)"
    )

    # -- Data setting
    parser.add_argument(
        "--content_path",
        type=str,
        default="./data/content/content.jpg",
        help="content iamge file name (default: ./data/content/content.jpg)",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default="./data/style/style.jpg",
        help="style image file name (default: ./data/style/style.jpg)",
    )

    # -- Model hyperparameters setting
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (defalut: 1000)",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="style_transfer",
        help="trainer type (default: style_transfer)",  # 향후 다른 모델을 구현했을 때, 학습 파이프라인 관리를 위해 수정 필요
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="content loss weight (default: 1)"
    )
    parser.add_argument(
        "--beta", type=float, default=1, help="style loss weight (default: 1)"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="log interval (default: 100)"
    )

    args = parser.parse_args()
    print(args)

    # -- save directory 초기화
    content_dir = args.content_path
    style_dir = args.style_path

    # -- trainer 설정
    trainer_name = args.trainer.lower()
    train_module = getattr(
        import_module(f"{trainer_name}_trainer"), args.trainer + "Trainer"
    )

    print(f" Use {trainer_name}_trainer ...")
    trainer = train_module(content_dir, style_dir, args)

    # -- 학습 + 시간 기록
    start_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 시작: {str(start_time)[:19]}")

    trainer.train(args)

    end_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 종료: {str(end_time)[:19]}")

    # -- 학습 소요 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"학습 소요 시간: {elapsed_time}")
