# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.densenet as dsn
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        memory_efficient=False
    ):

        super(Encoder, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                (
                    'conv0',
                    nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
                ),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = dsn._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers*growth_rate
            if i != len(block_config) - 1:
                trans = dsn._Transition(
                    num_input_features=num_features, num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.num_features = num_features

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return out


def deconv(in_channels, out_channels, k_size=5, stride=2, padding=2, bn=True):
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            output_padding=stride + 2*padding - k_size,
            bias=False
        )
    )
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, features_num, conv_dim=16, out_dim=1):
        super(Decoder, self).__init__()

        self.deconv1 = deconv(features_num, conv_dim*32, k_size=5, stride=2)  # 1x1 -> 2*2
        self.deconv2 = deconv(conv_dim*32, conv_dim*16, k_size=5, stride=2)  # 2x2 -> 4*4
        self.deconv3 = deconv(conv_dim*16, conv_dim*8, k_size=5, stride=2)  # 4x4 -> 8*8
        self.deconv4 = deconv(conv_dim*8, conv_dim*4, k_size=5, stride=2)  # 8x8 -> 16*16
        self.deconv5 = deconv(conv_dim*4, conv_dim*2, k_size=5, stride=2)  # 16*16 -> 32x32
        self.deconv6 = deconv(conv_dim*2, conv_dim*1, k_size=5, stride=2)  # 32x32 -> 64x64
        self.deconv7 = deconv(conv_dim*1, out_dim, k_size=5, stride=2, bn=False)  # 64x64 -> 128x128

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.leaky_relu(self.deconv1(x), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.leaky_relu(self.deconv4(out), 0.05)
        out = F.leaky_relu(self.deconv5(out), 0.05)
        out = F.leaky_relu(self.deconv6(out), 0.05)
        out = F.tanh(self.deconv7(out))
        return out


def conv(in_channels, out_channels, k_size=4, stride=4, bn=True):
    layers = []
    layers.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            stride=stride,
            padding=(k_size-stride) // 2,
            bias=False,
        )
    )
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = conv(in_channels, conv_dim*1, bn=False)  # 128->32, 1->64
        self.conv2 = conv(conv_dim*1, conv_dim*2)  # 32->8, 64->128
        self.conv3 = conv(conv_dim*2, conv_dim*4)  # 8->2, 128->256,
        self.conv4 = conv(conv_dim*4, conv_dim*8, stride=2, bn=False)  # 2->1, 256->512,

        self.bn = nn.BatchNorm2d(conv_dim*8)

        self.classifier = nn.Linear(conv_dim*8, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
