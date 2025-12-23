from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_classifier(
    net: nn.Module,
    trainloader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 0.02,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> float:
    """Generic supervised training loop returning the average loss."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    net.train()

    running_loss = 0.0
    batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
    return running_loss / max(batches, 1)


def evaluate_classifier(
    net: nn.Module,
    testloader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy) on the provided loader."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total else 0.0
    loss = loss / len(testloader) if len(testloader) else 0.0
    return loss, accuracy
