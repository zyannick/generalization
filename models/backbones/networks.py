# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

#from domainbed.lib import wide_resnet
import copy
from torch.autograd import Function
from models.backbones.i3d.pytorch_i3d import InceptionI3d
import models.backbones.stam.models as stam
from models.backbones.x3d import create_x3d_model


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torch.nn.init as init
import torch
from collections import OrderedDict
import torchvision.models as models_tv
from .i3d.pytorch_i3d import Unit3D


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


'''class feature_extractor(nn.Module):

    def __init__(self):
        super(feature_extractor, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_relu1', nn.ReLU())
        self.feature.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU())
        self.feature.add_module('f_conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.feature.add_module('f_relu3', nn.ReLU())
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.initialize_params()

    def forward(self, input_data):
        features = self.feature(input_data)

        return features

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()'''


class task_classifier(nn.Module):

    def __init__(self, spatial_squeeze = True, dropout_keep_prob=0.5, num_classes = 8):
        super(task_classifier, self).__init__()
        self._spatial_squeeze = spatial_squeeze
        self.dropout_keep_prob = dropout_keep_prob
        self._num_classes = num_classes
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits_task_classifier')
    def forward(self, x):
        logits = self.logits(self.dropout(x))
        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)
        return logits



class domain_discriminator(nn.Module):

    def __init__(self, rp_size, optimizer, lr, momentum, weight_decay, n_outputs=512):
        super(domain_discriminator, self).__init__()
        rp_size = 512
        n_outputs = 1024
        fl = 3 #512

        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(rp_size, 512))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU())
        self.domain_discriminator.add_module('d_drop1', nn.Dropout(0.2))

        self.domain_discriminator.add_module('d_fc2', nn.Linear(512, 256))
        self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc3', nn.Linear(256, 2))
        self.domain_discriminator.add_module('d_sfmax', nn.LogSoftmax(dim=1))
        # self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        # self.domain_discriminator.add_module('d_drop2', nn.Dropout())
        # self.domain_discriminator.add_module('d_fc3', nn.Linear(1024, 1))

        self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)

        self.initialize_params()

        # TODO Check the RP size
        self.projection = nn.Linear(n_outputs, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, input_data):
        # reverse_feature = ReverseLayer.apply(input_data, alpha)	# Make sure there will be no problem when updating discs params
        #print('on est dans forward')
        #print(input_data.shape)
        feature = input_data.view(input_data.size(0), -1)
        #print(feature.shape)
        feature_proj = self.projection(feature)
        #print(feature_proj.shape)

        domain_output = self.domain_discriminator(feature_proj)
        #print(domain_output.shape)

        #print('\n\n\n')

        return domain_output

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class domain_discriminator_ablation_RP(nn.Module):

    def __init__(self, optimizer, lr, momentum, weight_decay):
        super(domain_discriminator_ablation_RP, self).__init__()
        n_outputs = 512
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(n_outputs, 512))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU())
        self.domain_discriminator.add_module('d_drop1', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc2', nn.Linear(512, 256))
        self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc3', nn.Linear(256, 2))
        # self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d1_sfmax', nn.LogSoftmax(dim=1))

        self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)

        self.initialize_params()

    def forward(self, input_data):
        feature = input_data.view(input_data.size(0), -1)
        domain_output = self.domain_discriminator(feature)

        return domain_output

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()



def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams, flags):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    else:
        if flags.type_input == 'image':
            if input_shape[1:3] == (28, 28):
                return MNIST_CNN(input_shape)
            elif input_shape[1:3] == (224, 224):
                return ResNet(input_shape, hparams)
            else:
                raise NotImplementedError
        else:
            if flags.backbone == 'i3d':
                if flags.mode == 'rgb':
                    i3d = InceptionI3d(400, in_channels=3)
                    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'), strict=False)
                else:
                    i3d = InceptionI3d(400, in_channels=2)
                    i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
                i3d.replace_logits(flags.num_classes)
                return i3d
            elif flags.backbone == 'r2p1d':
                # From https://github.com/moabitcoin/ig65m-pytorch
                TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"
                base_model = 'ig65m'
                MODELS = {
                    # model: output classes
                    'r2plus1d_34_32_ig65m': 359,
                    'r2plus1d_34_32_kinetics': 400,
                    'r2plus1d_34_8_ig65m': 487,
                    'r2plus1d_34_8_kinetics': 400,
                }
                model_name = "r2plus1d_34_{}_{}".format(32, base_model)

                r2p1d = torch.hub.load(TORCH_R2PLUS1D,
                                    model_name,
                                    num_classes=MODELS[model_name],
                                    pretrained=True)

                r2p1d.fc = nn.Linear(r2p1d.fc.in_features, 2)
                return  r2p1d
            elif 'stam' in flags.backbone:
                model = stam.create_model(flags)
                return model
            elif flags.backbone == 'x3d':
                model = create_x3d_model(flags)
                return model
            else:
                raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

class Dann_Bakc(nn.Module):
    def __init__(self, feature_extractor, lc, dc):
        super(Dann_Bakcbone,  self).__init__()
        self.feature_ectractor = feature_extractor
        self.classifier = lc
        self.f = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                #nn.MaxPool2d(2,2),
                nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #nn.Conv2d(32, 32, kernel_size=3,padding=1,stride=2),
                #nn.BatchNorm2d(32),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                #nn.MaxPool2d(2, 2),
                nn.Conv2d(50, 50, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(50),
                nn.ReLU(),

                #nn.Conv2d(32, 32, kernel_size=3,padding=1,stride=2),
                #nn.BatchNorm2d(32),
                #nn.ReLU(),
                #nn.Conv2d(32, 128, kernel_size=5,padding=2),
                #nn.AvgPool2d(7)
            )
        self.lc = nn.Sequential(
            nn.Linear(50*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        self.dc = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )
    def forward(self, x,alpha):
        x = self.f(x)
        x = x.view(-1, 50*4*4)
        y = GRL.apply(x, alpha)
        x = self.lc(x)
        y = self.dc(y)
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        return x, y




class Dann_Bakcbone(nn.Module):
    def __init__(self, feature_extractor):
        super(Dann_Bakcbone,  self).__init__()
        self.feature_ectractor = feature_extractor

        self.lc = nn.Sequential(
            nn.Linear(50*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        self.dc = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )
    def forward(self, x,alpha):
        x = self.feature_extractor(x)
        x = x.view(-1, 50*4*4)
        y = GRL.apply(x, alpha)
        x = self.lc(x)
        y = self.dc(y)
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        return x, y