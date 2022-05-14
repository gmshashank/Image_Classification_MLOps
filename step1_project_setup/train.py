import os
import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer


def train(args: argparse.Namespace):
    # print(args)
    seed = args.manual_seed

    print("main")


def main():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--manual_seed", type=int, default=9, help="Manual Seed")
    parser.add_Argument("--num_classes", type=int, default=10, help="Number of Classes")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
