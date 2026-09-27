import numpy as np
import pandas as pd


def summarize(predictions: pd.DataFrame) -> dict:
    work = predictions.assign(error=predictions.prediction - predictions.actual)
    metrics = {"observations": int(len(work)), "mae": float(work.error.abs().mean()),
               "rmse": float(np.sqrt(np.mean(work.error**2))), "bias": float(work.error.mean())}
    pairs = work.sort_values(["game_id", "team_id"], kind="mergesort").groupby("game_id")
    differentials = pairs.filter(lambda group: len(group) == 2).groupby("game_id").apply(
        lambda group: (group.iloc[0].prediction - group.iloc[1].prediction)
        - (group.iloc[0].actual - group.iloc[1].actual), include_groups=False
    )
    metrics["drtg_mae"] = metrics["mae"]  # opponent ORtg is the paired defensive target
    metrics["differential_mae"] = float(differentials.abs().mean()) if len(differentials) else None
    metrics["differential_rmse"] = float(np.sqrt(np.mean(differentials**2))) if len(differentials) else None
    metrics["seasons"] = {str(k): float(v) for k, v in work.groupby("season").error.apply(lambda x: x.abs().mean()).items()} if "season" in work else {}
    bins = [-1, 0, 2, 5, 10, 20, np.inf]
    labels = ["Game 1", "Games 1-2", "Games 3-5", "Games 6-10", "Games 11-20", "Games 21+"]
    work["bucket"] = pd.cut(work.prior_games, bins=bins, labels=labels)
    metrics["maturity_buckets"] = {str(k): {"n": int(len(g)), "mae": float(g.error.abs().mean())} for k, g in work.groupby("bucket", observed=False) if len(g)}
    return metrics
