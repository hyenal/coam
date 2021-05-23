import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def get_mean_var(x, EPS=1e-3):
    b, c, _, _ = x.size()
    mean_x = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    # Add in some noise so that the std is not taking the sqrt of 0 which then leads to nan 
    # in the derivative in some cases
    std_x = (x.view(b, c, -1) + torch.randn(x.view(b, c, -1).size()).to(x.device) * EPS)
    std_x = std_x.std(dim=2).clamp(min=EPS).view(b, c, 1, 1)

    return mean_x, std_x

class Identity(nn.Module):
    def forward(self, x):
        return x

class ResizeBlock(nn.Module):
    def forward(self, x):
        b, c, _, _ = x.size()
        return x.view(b,c)

class NormBlock(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x / torch.norm(x, dim=self.dim, keepdim=True).clamp(min=1e-4)

class MLPBlock(nn.Module):
    def __init__(self, input_nc, output_nc, bn=False):
        super().__init__()
        self.linear = nn.Linear(input_nc, output_nc, bias=False)
        self.activation = nn.ReLU()

        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm1d(output_nc)

    def forward(self, x):
        if self.use_bn:
            return self.bn(self.activation(self.linear(x)))
        return self.activation(self.linear(x))
    
class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, relu=True, bias=False):
        super().__init__()
        if relu:
            self.convblocks = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=2, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(output_nc),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=1, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(output_nc),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=1, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(output_nc)
            )
        else:
            self.convblocks = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=2, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(output_nc),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=1, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(output_nc),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=1, bias=bias)
            )
    
    def forward(self, x):
        return self.convblocks(x)

class ConvAdainBlocks(nn.Module):
    def __init__(self, input_nc, adain_input_nc, nf, nl=1, bias=False):
        super().__init__()
        self.conv1 = ConvAdainBlock(input_nc, nf*4, adain_input_nc=adain_input_nc, stride=1, bias=bias) # 16
        self.conv2 = ConvAdainBlock(nf*4, nf*2, adain_input_nc=adain_input_nc, stride=1, bias=bias) # 16

        self.conv_layers = []
        for i in range(0, nl):
            self.conv_layers.append(
                ConvAdainBlock(nf*2, nf*2, adain_input_nc=adain_input_nc, relu=False, stride=1, bias=bias)) # 16

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.conv3 = nn.Sequential(nn.Conv2d(nf*2, nf*2, stride=1, kernel_size=3, bias=bias, padding=1), nn.BatchNorm2d(nf*2))

    def forward(self, x, y):
        x1 = self.conv1(x, y)
        x2 = self.conv2(x1, y)

        for i in range(0, len(self.conv_layers)):
            x2 = self.conv_layers[i](x2, y)

        res = self.conv3(x2)
        return res

class ConvAdainBlock(nn.Module):
    def __init__(self, input_nc, output_nc, adain_input_nc=64, relu=True, stride=2, bias=False):
        super().__init__()
        self.relu = relu

        if relu:
            self.convblocks1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=stride, bias=bias))
            self.adainBN1 = AdainBlock(adain_input_nc, output_nc)
            self.mod1 = nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0, stride=stride, bias=bias)
            self.convblocks2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=stride, bias=bias)
                )
            self.adainBN2 = AdainBlock(adain_input_nc, output_nc)
            self.mod2 = nn.Conv2d(output_nc, output_nc, kernel_size=1, padding=0, stride=stride, bias=bias)
        else:
            self.convblocks1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=stride, bias=bias))
            self.adainBN1 = AdainBlock(adain_input_nc, output_nc)
            self.mod1 = nn.Conv2d(output_nc, output_nc, kernel_size=1, padding=0, stride=stride, bias=bias)

    def forward(self, x, style):
        x1 = self.convblocks1(x)
        x1 = self.adainBN1(x1, style) + self.mod1(x)

        if not(self.relu):
            return x1

        x2 = self.convblocks2(x1)
        x2 = self.adainBN2(x2, style) + self.mod2(x1)
        return x2

class AdainBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.A = nn.Linear(in_c, out_c * 2, bias=True)

    def forward(self, x, y, EPS=1e-10):
        b, c, _, _ = x.size()
        mean_x, std_x = get_mean_var(x)
        style = self.A(y).view(b, c * 2, 1, 1)
        style_w = style[:,:c,:,:]
        style_b = style[:,c:,:,:]

        result = style_w * (x - mean_x) / std_x + style_b
        return result.contiguous()

class ResNetAdainBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_ns, mid='upsample'):
        super().__init__()

        if mid == 'upsample':
            norm = nn.Upsample(scale_factor=2, mode='bilinear')
        elif mid == 'identity':
            norm = Identity()
        elif mid == 'downsample':
            norm = nn.AvgPool2d(kernel_size=2)
        else:
            raise Exception("normalisation not known.")

        self.adain1 = AdainBlock(ch_ns, ch_in)
        self.adain2 = AdainBlock(ch_ns, ch_out)

        self.ch_a = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                norm,
                nn.Conv2d(ch_in, ch_out, stride=1, padding=1, kernel_size=3)),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(ch_out, ch_out, stride=1, padding=1, kernel_size=3))]
        )

        self.ch_b = nn.Sequential(
            norm, 
            nn.Conv2d(ch_in, ch_out, stride=1, padding=0, kernel_size=1)
        )

    def forward(self, xv):
        x = xv[0]; v = xv[1]
        x_a = self.adain1(x, v)
        x_a = self.ch_a[0](x_a)
        x_a = self.adain2(x_a, v)
        x_a = self.ch_a[1](x_a)

        x_b = self.ch_b(x)
        return (x_a + x_b, v)

class ResNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=False):
        super().__init__()

        AvgPool = nn.AvgPool2d
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d

        if downsample:
            stride = 2
        else:
            stride = 1

        self.ch_a = nn.Sequential(
            Conv(ch_in, ch_out, stride=stride, padding=1, kernel_size=3, bias=False),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            Conv(ch_out, ch_out, stride=1, padding=1, kernel_size=3, bias=False),
            BatchNorm(ch_out),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(ch_out)
            )
        else:
            self.downsample = None

    def forward(self, x):
        x_a = self.ch_a(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x_a + x

class ResNetNoiseBlocks(nn.Module):
    def __init__(self, channels, norms, seed=0, ch_ns=10):
        super().__init__()
        assert(len(channels) == len(norms)+1)

        self.ch_ns = ch_ns

        blocks = []
        for i in range(0, len(norms)):
            blocks += [ResNetAdainBlock(channels[i], channels[i+1], ch_ns=ch_ns, mid=norms[i])]

        self.blocks = nn.Sequential(*blocks)

        self.style_vec = nn.Linear(ch_ns, ch_ns, bias=False)

        self.rng = np.random.RandomState(seed)

    def forward(self, x):
        B = x.size(0)
        rand_input = torch.Tensor(self.rng.randn(B,self.ch_ns)).to(x.device)

        v = self.style_vec(rand_input)

        return self.blocks((x, v))[0]

class ResNetStyleBlocks(nn.Module):
    def __init__(self, channels, norms, ch_ns=32):
        super().__init__()
        assert(len(channels) == len(norms)+1)

        blocks = []
        for i in range(0, len(norms)):
            blocks += [ResNetAdainBlock(channels[i], channels[i+1], ch_ns=ch_ns, mid=norms[i])]

        self.blocks = nn.Sequential(*blocks)
        self.style_vec = torchvision.models.resnet18(pretrained=False, num_classes=ch_ns)

    def forward(self, x, y):
        v = self.style_vec(y)
        return F.sigmoid(self.blocks((x, v))[0])
