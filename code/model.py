import torch
import torch.nn as nn
import torchvision


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        res = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)

        res = list(res.children())[:-1]
        _res = nn.Sequential(*res)

        self.backbone = _res

        # for param in self.backbone.parameters():
            # param.requires_grad = False

        self.conv3d = nn.Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.batchnorm = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)

        x = self.conv3d(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = x.view(x.size()[0], -1)

        fc_x = self.fc(x)

        return x, fc_x


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyModel(10)
    model.to(device)

    model.train()

    imgs = torch.rand((2, 3, 10, 128, 128)).to(device)

    print(model)
    print(model(imgs))

    # summary(model, input_size=(3, 224, 112))


if __name__ == '__main__':
    test()

