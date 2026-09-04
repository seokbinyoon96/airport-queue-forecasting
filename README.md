# Airport Terminal Passenger Queue Forecasting

Transformer-based framework for forecasting passenger queue length and waiting time
at departure gates (DG) and security checkpoints (SC), using historical facility
queue dynamics together with check-in island throughput.

## Files
- `encoder.py` — pre-norm Transformer encoder layer (LayerNorm → Facility multi-head
  self-attention → feed-forward network) that exposes its attention weights
- `model.py` — full model: input embedding (inverted linear projection into variate
  tokens, prepended global token, temporal context embedding) → stacked Transformer
  encoder layers → two facility-specific MLP prediction heads
- `train.py` — training / validation loop, checkpointing
- `experiment.py` — accuracy evaluation (row-wise MAE / RMSE) on a trained checkpoint

## Model overview
Following an iTransformer-style embedding, the input sequence is transposed and
linearly projected into facility-level variate tokens — departure gate queue length,
departure gate waiting time, security checkpoint queue length, security checkpoint
waiting time, and check-in island throughput — each with an 18-step history (T = 18).
A learnable global token is prepended to the variate token sequence, and a temporal
context embedding (day-of-week and hour-of-day) is added to every token, yielding
37 variate tokens + 1 global token (38 total).

This sequence is processed by L stacked Transformer encoder layers, each composed of
layer normalization, Facility multi-head self-attention (modeling interactions across
all facility tokens), and a feed-forward network. For prediction, the final-layer
global token is concatenated with the average-pooled and max-pooled representations
of the variate tokens, and the result is passed to two facility-specific MLP heads to
forecast the next 12 steps (S = 12) of queue length and waiting time at departure
gates and security checkpoints.

## Usage
```python
import torch
import torch.nn as nn
from model import Transformer
import train

model = Transformer(d_model=256, nhead=4, num_layers=3, dim_ff=1024, dropout=0.1)

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
