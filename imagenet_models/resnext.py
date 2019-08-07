import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNeXtBottleneck(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, inplanes, cardinality=32, bottleneck_width=4, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride,
                               groups=cardinality, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, group_width*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(group_width*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality, bottleneck_width, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0])
        self.layer2 = self._make_layer(block, layers[1], stride=2)
        self.layer3 = self._make_layer(block, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.cardinality*self.bottleneck_width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, ResNeXtBottleneck):
                m.bn3.weight.data.zero_()

    def _make_layer(self, block, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != self.cardinality * self.bottleneck_width * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, self.cardinality * self.bottleneck_width * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.cardinality * self.bottleneck_width * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, self.cardinality, self.bottleneck_width, stride, downsample))
        self.inplanes = self.cardinality * self.bottleneck_width * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.cardinality, self.bottleneck_width))
        self.bottleneck_width *= 2

        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        if lin < 1 and lout > -1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        if lin < 2 and lout > 0:
            x = self.layer1(x)
        if lin < 3 and lout > 1:
            x = self.layer2(x)
        if lin < 4 and lout > 2:
            x = self.layer3(x)
        if lin < 5 and lout > 3:
            x = self.layer4(x)
        if lout > 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


def ResNeXt101_32x4d():
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], cardinality=32, bottleneck_width=4)
