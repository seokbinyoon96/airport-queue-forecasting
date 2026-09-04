# Airport Queue Forecasting

Transformer-based model for predicting terminal congestion (queue length and waiting time)
at departure gates (DG) and security checkpoints (SC) from short-term operational history.

## Files

- `encoder.py` — pre-norm Transformer encoder layer that exposes its attention weights
- `model.py` — full model: token embeddings (value + type + source + time context) → encoder stack → dual prediction heads
- `train.py` — training / validation loop, checkpointing
- `experiment.py` — accuracy evaluation (row-wise MAE / RMSE / MAPE) on a trained checkpoint

## Model overview

Inputs are grouped into 5 variable types (BagDrop, DG queue length, DG waiting time,
SC queue length, SC waiting time), each with an 18-step history, plus day/hour context.
These are embedded and concatenated into 37 tokens + 1 global token (38 total), passed
through a stack of Transformer encoder layers, pooled (global + mean + max), and decoded
into 12-step-ahead predictions for DG and SC queue length / waiting time.

## Usage

```python
import torch
import torch.nn as nn
from model import Transformer
import train

model = Transformer(d_model=512, nhead=4, num_layers=3, dim_ff=1024, dropout=0.1)

# train_loader / val_loader: yield batches of
# (BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before, Day, Hour, targets)
# where targets = (DG_QL, DG_WT, SC_QL, SC_WT)
train.main(train_loader, val_loader, save_dir="./checkpoints")
```

```python
from experiment import evaluate_on_test

evaluate_on_test("./checkpoints/best_model.pt", test_loader)
```

Data loading / preprocessing code is not included here — plug in your own `DataLoader`
that yields batches in the format described above.
