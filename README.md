# PC-RTRNet

Research code for multi-temporal remote-sensing image restoration.

## Components

- `pcrtrnet/blocks.py`: convolution, downsampling, and skip-connected upsampling blocks.
- `train.py`: a reusable training-epoch utility with caller-supplied model, objective, and optimizer.
- `test.py`: numerical evaluation with caller-supplied model and metrics.

## Environment

Python 3.10 or later and PyTorch are required. Install a PyTorch build suitable for your computing environment.

## Basic usage

```python
import torch
from pcrtrnet.blocks import DoubleConv, Down, Up

x = torch.randn(1, 8, 32, 32)
skip = DoubleConv(8, 16)(x)
encoded = Down(16, 32)(skip)
decoded = Up(32, 16)(encoded, skip)
print(decoded.shape)
```

`train_one_epoch` accepts batches of `(inputs, targets)`. The supplied objective must return a scalar batch-mean loss. The caller controls device placement, optimization settings, scheduling, and checkpoint management.

`evaluate` accepts the same batch structure and a mapping of metric names to callables. Each callable returns one scalar per sample. The function returns dataset means and does not export images.

These utilities do not prescribe image scaling or metric definitions. Keep those conventions consistent in your application. Channel sizes in the example illustrate block usage only.

## Terms

No software license has been designated for this repository.
