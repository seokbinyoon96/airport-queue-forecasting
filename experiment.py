"""Accuracy evaluation (row-wise MAE / RMSE / MAPE) for the trained model."""

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from train import load_checkpoint_model

# denormalization ranges used to report metrics in original units
NORM_INFO = {
    "min_dg_ql": 0.0, "max_dg_ql": 387.0,
    "min_dg_wt": 0.0, "max_dg_wt": 72.3333,
    "min_sc_ql": 0.0, "max_sc_ql": 281.0,
    "min_sc_wt": 0.0, "max_sc_wt": 39.7167,
}

DG_ROW_LABELS = ["DG2-W", "DG2-E", "DG3-W", "DG3-E", "DG4-W", "DG4-E", "DG5-W", "DG5-E"]
SC_ROW_LABELS = ["SC2", "SC3", "SC4", "SC5"]


@torch.no_grad()
def eval_rowwise_mae_rmse_mape(model, loader, device, norm_info=NORM_INFO, eps=1e-9, desc="Evaluating"):
    model.eval()

    dg_ql_range = norm_info["max_dg_ql"] - norm_info["min_dg_ql"]
    dg_wt_range = norm_info["max_dg_wt"] - norm_info["min_dg_wt"]
    sc_ql_range = norm_info["max_sc_ql"] - norm_info["min_sc_ql"]
    sc_wt_range = norm_info["max_sc_wt"] - norm_info["min_sc_wt"]

    def init_stat(n_rows):
        return {
            "abs_sum": np.zeros(n_rows), "sq_sum": np.zeros(n_rows),
            "count": np.zeros(n_rows, dtype=np.int64),
            "ape_sum": np.zeros(n_rows), "ape_count": np.zeros(n_rows, dtype=np.int64),
        }

    stats_dg_ql, stats_dg_wt = init_stat(8), init_stat(8)
    stats_sc_ql, stats_sc_wt = init_stat(4), init_stat(4)

    def update_stats(stats, pred, true):
        err = pred - true
        abs_err, sq_err = err.abs(), err.pow(2)

        stats["abs_sum"] += abs_err.sum(dim=(0, 2)).cpu().numpy()
        stats["sq_sum"] += sq_err.sum(dim=(0, 2)).cpu().numpy()
        stats["count"] += np.full(pred.shape[1], pred.shape[0] * pred.shape[2], dtype=np.int64)

        nonzero_mask = true.abs() > eps
        ape = torch.zeros_like(abs_err)
        ape[nonzero_mask] = abs_err[nonzero_mask] / true.abs()[nonzero_mask]
        stats["ape_sum"] += ape.sum(dim=(0, 2)).cpu().numpy()
        stats["ape_count"] += nonzero_mask.sum(dim=(0, 2)).cpu().numpy().astype(np.int64)

    for batch in tqdm(loader, desc=desc, leave=False):
        (
            BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before,
            Day, Hour, targets,
        ) = batch
        DG_QL, DG_WT, SC_QL, SC_WT = targets

        BagDrop, DG_QL_before, DG_WT_before = BagDrop.to(device), DG_QL_before.to(device), DG_WT_before.to(device)
        SC_QL_before, SC_WT_before = SC_QL_before.to(device), SC_WT_before.to(device)
        Day, Hour = Day.to(device).long(), Hour.to(device).long()
        DG_QL, DG_WT, SC_QL, SC_WT = DG_QL.to(device), DG_WT.to(device), SC_QL.to(device), SC_WT.to(device)

        pred_DG_QL, pred_DG_WT, pred_SC_QL, pred_SC_WT = model(
            (BagDrop, DG_QL_before, DG_WT_before, SC_QL_before, SC_WT_before, Day, Hour)
        )

        update_stats(stats_dg_ql, pred_DG_QL * dg_ql_range + norm_info["min_dg_ql"], DG_QL * dg_ql_range + norm_info["min_dg_ql"])
        update_stats(stats_dg_wt, pred_DG_WT * dg_wt_range + norm_info["min_dg_wt"], DG_WT * dg_wt_range + norm_info["min_dg_wt"])
        update_stats(stats_sc_ql, pred_SC_QL * sc_ql_range + norm_info["min_sc_ql"], SC_QL * sc_ql_range + norm_info["min_sc_ql"])
        update_stats(stats_sc_wt, pred_SC_WT * sc_wt_range + norm_info["min_sc_wt"], SC_WT * sc_wt_range + norm_info["min_sc_wt"])

    def build_df(labels, stats):
        mae = stats["abs_sum"] / np.maximum(stats["count"], 1)
        rmse = np.sqrt(stats["sq_sum"] / np.maximum(stats["count"], 1))
        mape = (stats["ape_sum"] / np.maximum(stats["ape_count"], 1)) * 100.0

        df = pd.DataFrame({"Metric": labels, "MAE": mae, "RMSE": rmse, "MAPE": mape})
        avg_row = pd.DataFrame([{"Metric": "Avg", "MAE": df["MAE"].mean(), "RMSE": df["RMSE"].mean(), "MAPE": df["MAPE"].mean()}])
        return pd.concat([df, avg_row], ignore_index=True)

    return (
        build_df(DG_ROW_LABELS, stats_dg_ql),
        build_df(DG_ROW_LABELS, stats_dg_wt),
        build_df(SC_ROW_LABELS, stats_sc_ql),
        build_df(SC_ROW_LABELS, stats_sc_wt),
    )


def print_metric_table(df, title, digits=3):
    df_show = df.copy()
    for col in ["MAE", "RMSE", "MAPE"]:
        df_show[col] = df_show[col].map(lambda x: round(float(x), digits))
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(df_show.to_string(index=False))


def evaluate_on_test(checkpoint_path, test_loader, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_model(checkpoint_path, device=device)

    df_dg_ql, df_dg_wt, df_sc_ql, df_sc_wt = eval_rowwise_mae_rmse_mape(
        model, test_loader, device, desc="TEST Row-wise MAE/RMSE/MAPE"
    )

    print_metric_table(df_dg_ql, "TEST RESULTS - DG Queue Length")
    print_metric_table(df_dg_wt, "TEST RESULTS - DG Waiting Time")
    print_metric_table(df_sc_ql, "TEST RESULTS - SC Queue Length")
    print_metric_table(df_sc_wt, "TEST RESULTS - SC Waiting Time")

    avgs = [df.loc[df["Metric"] == "Avg"].iloc[0] for df in (df_dg_ql, df_dg_wt, df_sc_ql, df_sc_wt)]
    overall_df = pd.DataFrame([{
        "Metric": "Overall Avg",
        "MAE": np.mean([a["MAE"] for a in avgs]),
        "RMSE": np.mean([a["RMSE"] for a in avgs]),
        "MAPE": np.mean([a["MAPE"] for a in avgs]),
    }])
    print("\n===== OVERALL TEST AVERAGE =====")
    print(overall_df.to_string(index=False))

    return overall_df, (df_dg_ql, df_dg_wt, df_sc_ql, df_sc_wt)


if __name__ == "__main__":
    # Supply your own test DataLoader (batch format described in train.py).
    # test_loader = ...
    # evaluate_on_test("./checkpoints/best_model.pt", test_loader)
    raise SystemExit("Provide test_loader and call evaluate_on_test(checkpoint_path, test_loader).")
