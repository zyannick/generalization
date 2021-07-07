import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.video.resnet import VideoResNet

from .load_model import *

import torchvision




class MyModelFeatures(nn.Module):
    def __init__(self, pretrained_model):
        super(MyModelFeatures, self).__init__()
        # super().__init__()
        # super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features

        #self.features = nn.Sequential(*list(pretrained_model.children())[:-2])

    def forward(self, x):
        y = self.features(x)
        return y


class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(MyModel, self).__init__()
        # super().__init__()
        # super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.pretrained_model = pretrained_model

        self.features = nn.Sequential(*list(pretrained_model.children())[:-2])

    def forward(self, x):
        x = self.pretrained_model(x)
        return x



class Conv2Plus1DExtended(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1DExtended, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, midplanes//2, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False),
            nn.BatchNorm3d(midplanes//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes//2, out_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding=(1, 1, 1),
                      bias=False)
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)

class MyTwoStreamModelExtended(nn.Module):
    def __init__(self, pretrained_model):
        super(MyTwoStreamModelExtended, self).__init__()
        self.sobel_part_1 = Conv2Plus1D(1, 6, 3)
        self.laplace_part_1 = Conv2Plus1D(1, 6, 3)
        self.kirsch_part_1 = Conv2Plus1D(1, 6, 3)
        self.fusion_1 = Conv2Plus1DExtended(18, 3, 12)
        self.pretrained_model = pretrained_model

    def forward(self, x):
        x_sobel = x[:, 0, :, :, :]
        x_laplace = x[:, 1, :, :, :]
        x_kirsch = x[:, 2, :, :, :]

        (batch, t, h, w) = x_sobel.size()

        x_sobel = x_sobel.view(batch, 1, t, h, w)
        x_laplace = x_laplace.view(batch, 1, t, h, w)
        x_kirsch = x_kirsch.view(batch, 1, t, h, w)

        x_sobel = self.sobel_part_1(x_sobel)

        x_laplace = self.laplace_part_1(x_laplace)

        x_kirsch = self.kirsch_part_1(x_kirsch)

        x = torch.cat((x_sobel, x_laplace, x_kirsch), dim=1)

        x_fusion = self.fusion_1(x)

        x = self.pretrained_model(x_fusion)
        return x

class MyTwoStreamModel(nn.Module):
    def __init__(self, pretrained_model):
        super(MyTwoStreamModel, self).__init__()

        self.sobel_part_1 = Conv2Plus1D(3, 64, 128)

        self.laplace_part_1 = Conv2Plus1D(3, 64, 128)

        self.kirsch_part_1 = Conv2Plus1D(3, 64, 128)

        self.fusion_1 = Conv2Plus1D(192, 3, 48)

        self.pretrained_model = pretrained_model

        self.features = nn.Sequential(*list(pretrained_model.children())[:])

    def forward(self, x_sobel, x_laplace, x_kirsch):

        x_sobel = self.sobel_part_1(x_sobel)

        x_laplace = self.laplace_part_1(x_laplace)

        x_kirsch = self.kirsch_part_1(x_kirsch)

        x = torch.cat((x_sobel, x_laplace, x_kirsch), dim=1)

        x_fusion = self.fusion_1(x)

        #print(x_fusion.shape)
        x = self.pretrained_model(x_fusion)
        return x
