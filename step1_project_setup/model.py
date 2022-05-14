import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LitCIFAR10Model(pl.LightningModule):
    def __init__(self,input_shape,num_classes,learning_rate=2e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate=learning_rate

        self.conv1=nn.Conv2d(3,32,3,1)
        self.conv2=nn.Conv2d(32,32,3,1)
        self.conv3=nn.Conv2d(32,64,3,1)
        self.conv4=nn.Conv2d(64,64,3,1)

        self.pool1=nn.MaxPool2d(2)
        self.pool2=nn.MaxPool2d(2)

        n_sizes=self._get_cov_output)input_shape
        self.fc1=nn.Linear(n_sizes,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,num_classes)

    
    def _forward_features(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool1(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=self.pool2(F.relu(self.conv4(x)))
        return x

    def _get_cov_output(self,shape,batch_size):
        input=torch.autograd.Variable(torch.rand(batch_size,*shape))

        output_feat=self._forward_features(input)
        n_size=output_feat.data.view(batch_size,-1).size(1)
        return n_size
    
    def forward(self,x):
        x=self._forward_features(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.log_softmax(self.fc3(x),dim=1)
        return x 
    
    def training_step(self,batch,batch_idx):
        x,y=batch
        logits=self(x)
        loss=F.nll_loss(logits,y)

        preds=torch.argmax(logits,dim=1)
        acc=accuracy(preds,y)


    # def validation_step(self,batch,batch_idx):