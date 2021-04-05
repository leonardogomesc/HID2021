import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from collections import OrderedDict
from torchvision.models.resnet import Bottleneck


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        res = torchvision.models.resnet50(pretrained=True, progress=True)
        res = list(res.children())[:-1]
        _res = nn.Sequential(*res)

        self.backbone = _res

        # for param in self.backbone.parameters():
            # param.requires_grad = False

        self.conv2d = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
        self.batchnorm = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.backbone(x)

        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = F.relu(x)

        x = x.view(x.size()[0], -1)

        return x


class MyComplexModel(nn.Module):
    def __init__(self):
        super(MyComplexModel, self).__init__()
        res = torchvision.models.resnet50(pretrained=True, progress=True)
        res = list(res.children())[:-2]
        _res = nn.Sequential(*res)

        self.backbone = _res

        # for param in self.backbone.parameters():
            # param.requires_grad = False

        self.bottlenecks = nn.Sequential(Bottleneck(2048, 512),
                                         Bottleneck(2048, 512))

        self.global_stream = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1)),
                                           nn.BatchNorm2d(512),
                                           nn.ReLU())

        self.dropblock_stream = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1)),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU())

        self.regularization_stream = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                   nn.ReLU())

    def forward(self, imgs, masks, training_mask=False):
        if not self.training:
            resnet_features = self.backbone(imgs)

            bottleneck = self.bottlenecks(resnet_features)
            dropblock = self.dropblock_stream(bottleneck)
            dropblock = dropblock.view(dropblock.size()[0], -1)

            global_out = self.global_stream(resnet_features)
            global_out = global_out.view(global_out.size()[0], -1)

            return torch.cat((global_out, dropblock), 1)

        if training_mask:
            masked_imgs = imgs * masks
            resnet_features_masked = self.backbone(masked_imgs)

            with torch.no_grad():
                resnet_features = self.backbone(imgs)
            '''global_out = self.global_stream(resnet_features)
            global_out = global_out.view(global_out.size()[0], -1)

            return global_out'''
            return resnet_features_masked, resnet_features

        masked_imgs = imgs * masks

        resnet_features_masked = self.backbone(masked_imgs)
        bottleneck_masked = self.bottlenecks(resnet_features_masked)
        dropblock = self.dropblock_stream(bottleneck_masked)
        dropblock = dropblock.view(dropblock.size()[0], -1)

        resnet_features = self.backbone(imgs)

        global_out = self.global_stream(resnet_features)
        global_out = global_out.view(global_out.size()[0], -1)

        bottleneck = self.bottlenecks(resnet_features)
        regularization = self.regularization_stream(bottleneck)
        regularization = regularization.view(regularization.size()[0], -1)

        return global_out, dropblock, regularization


# from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
class MyUNetModel(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(MyUNetModel, self).__init__()

        features = init_features
        self.encoder1 = MyUNetModel._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MyUNetModel._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MyUNetModel._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = MyUNetModel._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MyUNetModel._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = MyUNetModel._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = MyUNetModel._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = MyUNetModel._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = MyUNetModel._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyComplexModel()
    model.to(device)

    model.train()

    imgs = torch.rand((2, 3, 224, 112)).to(device)
    masks = torch.rand((2, 1, 224, 112)).to(device)


    print(model)
    print(model(imgs, masks))

    # summary(model, input_size=(3, 224, 112))


if __name__ == '__main__':
    test()

