"""Training loop for the terminal complexity Transformer.

Expects each batch from the DataLoader as:
    (BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before, Day, Hour, targets)
where targets = (DG_QL, DG_WT, SC_QL, SC_WT).
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from model import Transformer

# model hyperparameters
D_MODEL = 512
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 1024
DROPOUT = 0.1
MAX_LEN_PE = 200

# training hyperparameters
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0


def _move_batch_to_device(batch, device):
    (
        BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before,
        Day, Hour, targets,
    ) = batch
    DG_QL, DG_WT, SC_QL, SC_WT = targets

    return (
        BagDrop.to(device),
        DG_QL_before.to(device),
        DG_WT_before.to(device),
        SC_QL_before.to(device),
        SC_WT_before.to(device),
        Day.to(device).long(),
        Hour.to(device).long(),
        DG_QL.to(device),
        DG_WT.to(device),
        SC_QL.to(device),
        SC_WT.to(device),
    )


def _compute_loss_and_outputs(model, batch_tensors, criterion):
    (
        BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before,
        Day, Hour, DG_QL, DG_WT, SC_QL, SC_WT,
    ) = batch_tensors

    dep_QL, dep_WT, sec_QL, sec_WT = model(
        (BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before, Day, Hour)
    )

    loss_dg_ql = criterion(dep_QL, DG_QL)
    loss_dg_wt = criterion(dep_WT, DG_WT)
    loss_sc_ql = criterion(sec_QL, SC_QL)
    loss_sc_wt = criterion(sec_WT, SC_WT)
    loss_total = 2 * loss_dg_ql + loss_dg_wt + 2 * loss_sc_ql + loss_sc_wt

    pred_all = torch.cat([
        dep_QL.reshape(dep_QL.size(0), -1),
        dep_WT.reshape(dep_WT.size(0), -1),
        sec_QL.reshape(sec_QL.size(0), -1),
        sec_WT.reshape(sec_WT.size(0), -1),
    ], dim=1)
    true_all = torch.cat([
        DG_QL.reshape(DG_QL.size(0), -1),
        DG_WT.reshape(DG_WT.size(0), -1),
        SC_QL.reshape(SC_QL.size(0), -1),
        SC_WT.reshape(SC_WT.size(0), -1),
    ], dim=1)

    return {
        "loss_total": loss_total,
        "loss_dg_ql": loss_dg_ql,
        "loss_dg_wt": loss_dg_wt,
        "loss_sc_ql": loss_sc_ql,
        "loss_sc_wt": loss_sc_wt,
        "pred_all": pred_all,
        "true_all": true_all,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None, grad_clip_norm=1.0):
    model.train()

    sums = {"total": 0.0, "DG_QL": 0.0, "DG_WT": 0.0, "SC_QL": 0.0, "SC_WT": 0.0}
    abs_err_sum, elem_count = 0.0, 0

    pbar = tqdm(loader, desc=f"Train | Epoch {epoch}", leave=False)
    for batch in pbar:
        batch_tensors = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = _compute_loss_and_outputs(model, batch_tensors, criterion)
        out["loss_total"].backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        sums["total"] += out["loss_total"].item()
        sums["DG_QL"] += out["loss_dg_ql"].item()
        sums["DG_WT"] += out["loss_dg_wt"].item()
        sums["SC_QL"] += out["loss_sc_ql"].item()
        sums["SC_WT"] += out["loss_sc_wt"].item()

        abs_err_sum += torch.abs(out["pred_all"] - out["true_all"]).sum().item()
        elem_count += out["pred_all"].numel()

        pbar.set_postfix(mae=f"{abs_err_sum / max(elem_count, 1):.6f}")

    n_batches = len(loader)
    return {
        **{k: v / n_batches for k, v in sums.items()},
        "mae": abs_err_sum / max(elem_count, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=None):
    model.eval()

    sums = {"total": 0.0, "DG_QL": 0.0, "DG_WT": 0.0, "SC_QL": 0.0, "SC_WT": 0.0}
    abs_err_sum, elem_count = 0.0, 0

    pbar = tqdm(loader, desc=f"Val   | Epoch {epoch}", leave=False)
    for batch in pbar:
        batch_tensors = _move_batch_to_device(batch, device)
        out = _compute_loss_and_outputs(model, batch_tensors, criterion)

        sums["total"] += out["loss_total"].item()
        sums["DG_QL"] += out["loss_dg_ql"].item()
        sums["DG_WT"] += out["loss_dg_wt"].item()
        sums["SC_QL"] += out["loss_sc_ql"].item()
        sums["SC_WT"] += out["loss_sc_wt"].item()

        abs_err_sum += torch.abs(out["pred_all"] - out["true_all"]).sum().item()
        elem_count += out["pred_all"].numel()

        pbar.set_postfix(mae=f"{abs_err_sum / max(elem_count, 1):.6f}")

    n_batches = len(loader)
    return {
        **{k: v / n_batches for k, v in sums.items()},
        "mae": abs_err_sum / max(elem_count, 1),
    }


def convert_old_state_dict_to_attention_model(old_state_dict):
    new_state_dict = {}
    for k, v in old_state_dict.items():
        new_key = k.replace("encoder.layers.", "encoder_layers.") if k.startswith("encoder.layers.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def run_training(model, train_loader, val_loader, optimizer, criterion, device, save_dir, num_epochs=1000):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / "best_model.pt"
    last_model_path = save_dir / "last_model.pt"
    model_kwargs = {
        "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF, "dropout": DROPOUT, "max_len_pe": MAX_LEN_PE,
    }

    best_val_mae = float("inf")
    history = []

    epoch_bar = trange(1, num_epochs + 1, desc="Training", leave=True)
    for epoch in epoch_bar:
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, GRAD_CLIP_NORM)
        val_log = validate(model, val_loader, criterion, device, epoch)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_log.items()},
        })

        if val_log["mae"] < best_val_mae:
            best_val_mae = val_log["mae"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mae": best_val_mae,
                "model_kwargs": model_kwargs,
            }, best_model_path)
            print(f"[SAVED] {best_model_path} (epoch={epoch}, best_val_mae={best_val_mae:.8f})")

        epoch_bar.set_postfix(
            train_total=f"{train_log['total']:.6f}",
            val_total=f"{val_log['total']:.6f}",
            val_mae=f"{val_log['mae']:.6f}",
            best_mae=f"{best_val_mae:.6f}",
        )

    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": best_val_mae,
        "model_kwargs": model_kwargs,
    }, last_model_path)
    print(f"[SAVED] {last_model_path} (final epoch weights)")

    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    return best_model_path, last_model_path, history_df


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_kwargs = checkpoint.get("model_kwargs", {
        "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF, "dropout": DROPOUT, "max_len_pe": MAX_LEN_PE,
    })
    model = Transformer(**model_kwargs).to(device)

    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("encoder.layers.") for k in state_dict.keys()):
        state_dict = convert_old_state_dict_to_attention_model(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[LOADED] {checkpoint_path} (epoch={checkpoint.get('epoch', 'N/A')}, "
          f"best_val_mae={checkpoint.get('best_val_mae', float('nan')):.8f})")
    return model, checkpoint


def main(train_loader, val_loader, save_dir="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = Transformer(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_ff=DIM_FF, dropout=DROPOUT, max_len_pe=MAX_LEN_PE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    return run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        num_epochs=NUM_EPOCHS,
    )


if __name__ == "__main__":
    # Supply your own DataLoaders (batch format described at the top of this file).
    # train_loader, val_loader = ...
    # main(train_loader, val_loader, save_dir="./checkpoints")
    raise SystemExit("Provide train_loader / val_loader and call main(train_loader, val_loader).")
