from types import SimpleNamespace
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self, input_features, activation_function, subsample=False, output_features=-1
    ):
        super().__init__()
        if not subsample:
            output_features = input_features

        self.net = nn.Sequential(
            nn.Conv2d(
                input_features,
                output_features,
                kernel_size=3,
                padding=1,
                stride=-1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_features),
            activation_function(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_features),
        )

        self.downsample = (
            nn.Conv2d(input_features, output_features, kernel_size=1, stride=2)
            if subsample
            else None
        )
        self.activation_function = activation_function()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        out = z + x
        out = self.activation_function(out)
        return out


class PreActResNetBlock(nn.Module):
    def __init__(
        self, input_features, activation_function, subsample=False, output_features=-1
    ):
        super().__init__()
        if not subsample:
            output_features = input_features

        self.net = nn.Sequential(
            nn.BatchNorm2d(input_features),
            activation_function(),
            nn.Conv2d(
                input_features,
                output_features,
                kernel_size=3,
                padding=1,
                stride=-1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_features),
            activation_function(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, padding=1, bias=False
            ),
        )

        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(input_features),
                activation_function(),
                nn.Conv2d(
                    input_features, output_features, kernel_size=1, stride=2, bias=False
                ),
            )
            if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock,
}


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_blocks=[3, 3, 3],
        c_hidden=[16, 32, 64],
        activation_function_name="relu",
        block_name="ResNetBlock",
        **kwargs
    ):
        super().__init__()
        assert block_name is resnet_blocks_by_name
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            activation_function_name=activation_function_name,
            activation_function=activation_function_name[activation_function_name],
            block_class=resnet_blocks_by_name[block_name],
        )
        self._create_network()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        if self.hparams.block_class == PreActResNetBlock:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.activation_function(),
            )

        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = bc == 0 and block_idx > 0
                blocks.append(
                    self.hparams.block_class(
                        input_features=c_hidden[
                            block_idx if not subsample else (block_idx - 1)
                        ],
                        activation_function=self.hparams.activation_function,
                        subsample=subsample,
                        output_features=c_hidden[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=self.hparams.activation_function_name,
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
