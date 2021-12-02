import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, embedding_dim, logits=False, number_classes=None):
        super(ResNet, self).__init__()

        self.logits = logits

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.embedding_layer = nn.Linear(512*block.expansion, embedding_dim)

        if self.logits:
            assert isinstance(number_classes, int), "Number of classes must be provided when using logits"
            self.logits_layer = nn.Linear(embedding_dim, number_classes)
    

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        embeddings = self.embedding_layer(out)

        if self.logits:
            logits = self.logits_layer(F.relu(embeddings))
            return embeddings, logits
        
        return embeddings


def CifarResNet18(embedding_dim, logits=False, number_classes=None):
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    return ResNet(BasicBlock, [2, 2, 2, 2], embedding_dim, logits, number_classes)


class ResnetImageEncoder(nn.Module):
    def __init__(self, embedding_dim, resnet_size=50, pretrained=False, logits=False, number_classes=None):
        assert resnet_size in [18, 50, 101], "Resnet size must be either 18, 50, 101"
        super().__init__()

        if resnet_size==18:
            backbone = torchvision.models.resnet18(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(512, embedding_dim)
        elif resnet_size==50:
            backbone = torchvision.models.resnet50(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(2048, embedding_dim)
        elif resnet_size==101:
            backbone = torchvision.models.resnet101(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(2048, embedding_dim)

        self.logits = logits

        if self.logits:
            assert isinstance(number_classes, int), "Number of classes must be provided when using logits"
            self.logits_layer = nn.Linear(embedding_dim, number_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        embeddings = self.backbone(x)

        if self.logits:
            logits = self.logits_layer(self.relu(embeddings))
            return embeddings, logits
        
        return embeddings


class MlpImageEncoder(nn.Module):

    def __init__(self, embedding_dim, input_size, hidden_dim=256, n_internal_layers=6, dropout=0.3, logits=False, number_classes=None):
        
        super().__init__()

        self.logits = logits
        self.input_relu = nn.ReLU()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.inner_layer = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_internal_layers)])
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        
        if logits:
            assert isinstance(number_classes, int), "Number of classes must be provided when using logits"
            self.logits_layer = nn.Linear(embedding_dim, number_classes)
            self.embedding_relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_relu(self.fc1(x))
        
        for layer in self.inner_layer:
            x = layer(x)

        embedding = self.embedding_layer(x)

        if self.logits:
            logits = self.logits_layer(self.embedding_relu(embedding))
            return embedding, logits

        return embedding


class LeNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.3, logits=False, number_classes=None):
        super().__init__()

        # our in channel here are 1 because of our input size 1x28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1
        )

        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(256, 512)  
        self.fc2 = nn.Linear(512, 512)

        self.embedding_layer = nn.Linear(512, embedding_dim)

        self.logits = logits
        if logits:
            assert isinstance(number_classes, int), "Number of classes must be provided when using logits"
            self.logits_layer = nn.Linear(embedding_dim, number_classes)
            self.embedding_relu = nn.ReLU()

        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):

        x = self.pool1(self.activation(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout3(x)
        x = self.activation(self.fc2(x))

        embedding = self.embedding_layer(x)

        if self.logits:
            logits = self.logits_layer(self.embedding_relu(embedding))
            return embedding, logits

        return embedding
