import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


# class LitCIFAR10Model(pl.LightningModule):
#     def __init__(self, input_shape, num_classes, learning_rate=2e-4):
#         super().__init__()

#         self.save_hyperparameters()
#         self.learning_rate = learning_rate

#         self.conv1 = nn.Conv2d(3, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 32, 3, 1)
#         self.conv3 = nn.Conv2d(32, 64, 3, 1)
#         self.conv4 = nn.Conv2d(64, 64, 3, 1)

#         self.pool1 = nn.MaxPool2d(2)
#         self.pool2 = nn.MaxPool2d(2)

#         n_sizes = self._get_cov_output(input_shape)
#         self.fc1 = nn.Linear(n_sizes, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#     def _forward_features(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.pool2(F.relu(self.conv4(x)))
#         return x

#     def _get_cov_output(self, shape, batch_size):
#         input = torch.autograd.Variable(torch.rand(batch_size, *shape))

#         output_feat = self._forward_features(input)
#         n_size = output_feat.data.view(batch_size, -1).size(1)
#         return n_size

#     def forward(self, x):
#         x = self._forward_features(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x), dim=1)
#         return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)

#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy(preds, y)

#     # def validation_step(self,batch,batch_idx):


def create_model(model_dict, model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'


class CIFARModule(pl.LightningModule):
    def __init__(
        self, model_dict, model_name, model_hparams, optimizer_name, optimizer_hparams
    ):
        super().__init__()
        self.sav_hyperparameters()
        self.model = create_model(model_dict, model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}" '
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestone=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, baych, batch_idx):
        imgs, labels = batch_idx
        preds = self.model(imgs)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc)
