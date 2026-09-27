"""Idempotent, provenance-recorded single-file synchronizer."""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cbb.governance.sealed import assert_unsealed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    assert_unsealed([args.season])
    with urllib.request.urlopen(args.url, timeout=120) as response:  # noqa: S310
        payload = response.read()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    rows = max(payload.count(b"\n") - 1, 0) if destination.suffix == ".csv" else None
    manifest = {
        "dataset": args.dataset,
        "season": args.season,
        "rows": rows,
        "source": args.url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0.0",
    }
    path = Path("data/manifests") / f"{args.dataset}_{args.season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pulled": 1, "inserted": 1, "updated": 0, "rejected": 0}))


if __name__ == "__main__":
    main()
