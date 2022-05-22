from gc import callbacks
import os
import argparse
import numpy as np
import random
import torch
from torch.utils import data
from torchvision.datasets import CIFAR10
from torchvision import transforms

import pytorch_lightning as pl
# from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint

from step1_project_setup.model import CIFARModule

DATASET_PATH = "../data"
CHECKPOINT_PATH="../saved_models"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_CIFAR10_data(args: argparse.Namespace):
    seed = args.manual_seed

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, downlaod=True)
    data_mean = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    data_std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    print(f"Data Mean : {data_mean}")
    print(f"Data std. dev. : {data_std}")
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(data_mean, data_std)]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.totensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    )

    train_dataset = CIFAR10(
        root=DATASET_PATH, train=True, transform=train_transform, download=True
    )
    val_dataset = CIFAR10(
        root=DATASET_PATH, train=True, transform=test_transform, download=True
    )
    set_seed(seed)
    train_set, _ = data.random_split(train_dataset, [45000, 5000])
    set_seed(seed)
    _, val_set = data.random_split(val_dataset, [45000, 5000])
    test_set = CIFAR10(
        root=DATASET_PATH, train=False, transform=test_transform, download=True
    )

    train_loader = data.DataLoader(
        train_set,
        batch=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return train_loader,val_loader,test_loader

def train_model(model_name,args: argparse.Namespace,save_name=None,**kwargs):
    if save_name is None:
        save_name=model_name
    
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    num_gpus=1 if str(device)=="cuda:0" else 0

    trainer=pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH,save_name),
        gpus=num_gpus,
        max_epochs=args.num_epochs,
        callbacks=[ModelCheckpoint(save_weights_only=True,mode="max",monitor="val_acc"),LearningRateMonitor("epoch")],
        progress_bar_refresh_rate=1
    )
    trainer.logger._log_graph=True
    trainer.logger._default_hp_metric=None
    
    train_loader,val_loader,test_loader=setup_CIFAR10_data(args)

    pretrained_filename=os.path.join(CHECKPOINT_PATH,save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Pretrained Model found. Loading the model: {pretrained_filename}")
        model=CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(args.manual_seed)
        model=CIFARModule(model_name=model_name,**kwargs)
        trainer.fit(model,train_loader,val_loader)
        model=CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    val_result=trainer.test(model,val_loader,verbose=False)
    test_result=trainer.test(model,test_loader,verbose=False)
    result={"test":test_result[0]["test_acc"],"val":val_result[0]["test_acc"]}

    return model, result


def train(args: argparse.Namespace):
    # print(args)
    seed = args.manual_seed
    pl.seed_everything(seed)

    print("main")
    model_dict={}


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_Argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--manual_seed", type=int, default=9, help="Manual Seed")
    parser.add_Argument("--num_classes", type=int, default=10, help="Number of Classes")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of Workers")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
