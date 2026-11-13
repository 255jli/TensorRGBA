import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarGenerator(nn.Module):
    def __init__(self, noise_dim=128, num_classes=2):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(noise_dim + num_classes, 128 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def forward(self, noise, labels):
        labels = F.one_hot(labels, self.num_classes).float()
        x = torch.cat([noise, labels], dim=1)
        x = self.fc(x).view(-1, 128, 4, 4)
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        return torch.tanh(self.deconv3(x))

    def export_tensors(self):
        return [
            self.fc.weight,
            self.fc.bias,
            self.bn0.weight,
            self.bn0.bias,
            self.bn0.running_mean,
            self.bn0.running_var,
            self.deconv1.weight,
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.deconv2.weight,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.deconv3.weight,
            self.deconv3.bias,
        ]


class CifarDiscriminator(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(3 + num_classes, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
        )

    def forward(self, images, labels):
        labels = F.one_hot(labels, self.num_classes).float()
        label_map = labels[:, :, None, None].expand(
            -1, -1, images.size(2), images.size(3)
        )
        return self.layers(torch.cat([images, label_map], dim=1)).flatten(1)


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.zeros_(layer.bias)