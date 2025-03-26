import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from copy import deepcopy
import numpy as np
import ai3


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def conv2d():
    dataset = datasets.MNIST(
        root='./datasets', train=True, download=True,
        transform=transforms.ToTensor())
    subset_size = 1000
    indices = np.random.choice(
        len(dataset), subset_size, replace=False).tolist()
    dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset, batch_size=10, shuffle=True)

    model = ConvNet()
    model_swapped = deepcopy(model)
    ai3.swap_conv2d(model_swapped)

    lr = 0.001
    atol = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(), lr=lr)
    optimizer_swapped = Adam(
        model_swapped.parameters(), lr=lr)

    model.train()
    model_swapped.train()

    for epoch in range(5):
        total_loss_model = 0.0
        total_loss_swapped = 0.0
        for (inputs, labels) in loader:
            optimizer.zero_grad()
            optimizer_swapped.zero_grad()

            outputs = model(inputs)
            outputs_swapped = model_swapped(
                inputs)

            loss = criterion(outputs, labels)
            loss_swapped = criterion(
                outputs_swapped, labels)

            loss.backward()
            loss_swapped.backward()

            for param, param_swapped in zip(
                    model.named_parameters(),
                    model_swapped.named_parameters()):
                assert (param[1].grad is not None)
                assert (param_swapped[1].grad is not None)
                if not torch.allclose(
                        param[1].grad, param_swapped[1].grad, atol=atol):
                    print(
                        f'Grads differ for {param[0]}: {param[1].grad} vs {param_swapped[1].grad}')
                    exit(1)

            optimizer.step()
            optimizer_swapped.step()

            for param, param_swapped in zip(
                    model.named_parameters(),
                    model_swapped.named_parameters()):
                if not torch.allclose(param[1], param_swapped[1], atol=atol):
                    print(
                        f'Weights differ for {param[0]}: {param[1]} vs {param_swapped[1]}')
                    exit(1)

            total_loss_model += loss.item()
            total_loss_swapped += loss_swapped.item()

        print(
            f'Epoch {epoch+1} - Loss Orig: {total_loss_model / len(loader):.4f}')
        print(
            f'Epoch {epoch+1} - Loss Swapped: {total_loss_swapped / len(loader):.4f}')

    model_loss = evaluate_loss(
        model, loader, criterion)
    swapped_loss = evaluate_loss(
        model_swapped, loader, criterion)
    final_atol = 1e-3
    if abs(model_loss - swapped_loss) > final_atol:
        print(
            f'Final loss for orig {model_loss} and swapped {swapped_loss} differ')
    else:
        print(
            f'Final loss for orig and swapped are within atol {final_atol}')


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(
                outputs.data, 1)
            total += labels.size(0)
            correct += (predicted ==
                        labels).sum().item()

    avg_loss = total_loss / len(loader)
    return avg_loss
