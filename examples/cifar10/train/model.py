import torch
import torch.nn as nn


class AnimalGenerator(nn.Module):
    def __init__(self, noise_dim=128, num_classes=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(noise_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
