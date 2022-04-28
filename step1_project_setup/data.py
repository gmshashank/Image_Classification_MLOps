from matplotlib import transforms
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms,datasets
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,batch_size:int,num_classes:int=10,data_dir:str="./"):
        super().__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.dims=(3,32,32)
        self.num_classes=num_classes

        self.transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ]
        )
        
    def prepare_data(self):
        # Download CIFAR data
        datasets.CIFAR10(self.data_dir,train=True,download=True)
        datasets.CIFAR10(self.data_dir,train=False,download=True)
    
    def setup(self,stage:str=None):
        # Assign train/val datasets for use in dataloaders
        if stage=="fit" or stage is None:
            cifar_full=datasets.CIFAR10(self.data_dir,train=True,transform=self.transform)
            self.cifar_train,self.cifar_val=random_split(cifar_full,[45000,5000])
        
        if stage=="test" or stage is None:
            self.cifar_test=datasets.CIFAR10(self.data_dir,train=False,transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train,batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val,batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test,batch_size=self.batch_size)

if __name__=="__main__":
    cifar_datamodule=CIFAR10DataModule()
    cifar_datamodule.prepare_data()
    cifar_datamodule.setup()
    print(next(iter(cifar_datamodule.train_dataloader)))