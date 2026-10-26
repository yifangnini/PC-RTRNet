"""Reusable training utilities."""
import torch


def train_one_epoch(model, loader, optimizer, objective, device):
    """Train on tensor pairs using a scalar batch-mean objective."""
    model.train()
    total, count = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(inputs), targets)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("The objective must return a finite scalar.")
        loss.backward()
        optimizer.step()
        size = targets.shape[0]
        total += loss.detach().item() * size
        count += size
    if count == 0:
        raise ValueError("The training loader is empty.")
    return total / count
