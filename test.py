"""Numerical evaluation utilities."""
import torch


@torch.no_grad()
def evaluate(model, loader, metrics, device):
    """Average per-sample metrics; each callable returns a tensor of shape [B]."""
    if not metrics:
        raise ValueError("Supply at least one metric.")
    previous_mode = model.training
    totals = {name: 0.0 for name in metrics}
    count = 0
    model.eval()
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            size = targets.shape[0]
            for name, metric in metrics.items():
                values = metric(predictions, targets)
                if values.shape != (size,):
                    raise ValueError(f"Metric {name} must return one value per sample.")
                if not torch.isfinite(values).all().item():
                    raise ValueError(f"Metric {name} returned non-finite values.")
                totals[name] += values.double().sum().item()
            count += size
    finally:
        model.train(previous_mode)
    if count == 0:
        raise ValueError("The evaluation loader is empty.")
    return {name: total / count for name, total in totals.items()}
