import torch
import torch.nn as nn
import torchvision


class ResnetImageEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256, resnet_size=50, pretrained=True, logits=False, number_classes=None):
        assert resnet_size in [18, 50, 101], "Resnet size must be either 18, 50, 101"
        super().__init__()

        if resnet_size==18:
            backbone = torchvision.models.resnet18(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(512, hidden_dim)
        elif resnet_size==50:
            backbone = torchvision.models.resnet50(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(2048, hidden_dim)
        elif resnet_size==101:
            backbone = torchvision.models.resnet101(pretrained=pretrained, progress=False)
            backbone.fc = nn.Linear(2048, hidden_dim)

        self.backbone = backbone
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        self.logits = logits

        if self.logits:
            assert isinstance(number_classes, int), "Number of classes must be provided when using logits"
            self.logits_layer = nn.Linear(embedding_dim, number_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gelu(self.backbone(x))
        embeddings = self.embedding_layer(x)

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


class MnistCnnEncoder(nn.Module):
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
