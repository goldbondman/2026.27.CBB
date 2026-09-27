"""Canonical command-line backtester."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from cbb.backtest.metrics import summarize
from cbb.governance.sealed import assert_unsealed
from cbb.ratings.recursive import RecursiveOD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Canonical parquet or CSV")
    parser.add_argument("--output", default="results/run")
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = yaml.safe_load(config_bytes)
    assert_unsealed(config["seasons"])
    source = Path(args.input)
    data_bytes = source.read_bytes()
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    assert_unsealed(frame.season.unique())
    updater = config["updating"]
    model = RecursiveOD(base_k=config["base_k"], maturity_amplitude=updater["maturity_amplitude"],
                        maturity_horizon=updater["maturity_horizon_games"],
                        multiplier_min=updater["multiplier_min"], multiplier_max=updater["multiplier_max"])
    predictions = model.run(frame)
    predictions = predictions.merge(frame[["game_id", "team_id", "season"]], on=["game_id", "team_id"], how="left", validate="one_to_one")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.12g")
    record = {"experiment_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
              "timestamp": datetime.now(timezone.utc).isoformat(), "model_status": "FROZEN",
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "data_sha256": hashlib.sha256(data_bytes).hexdigest(), "metrics": summarize(predictions),
              "leakage_status": "tested_by_suite", "determinism_status": "stable_sort_and_serialization"}
    (output / "experiment.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

