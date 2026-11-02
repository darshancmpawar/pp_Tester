"""
Offline training CLI — builds per-pax model artifacts for one or all clients.

Used by .github/workflows/train-models.yml. The Streamlit app refuses to
train at runtime; this script is the only producer of artifacts/ files.

Usage:
    python scripts/train.py                       # train all embedding clients
    python scripts/train.py --client tekion       # train one client
    python scripts/train.py --client all --force  # retrain even if present

Exit codes:
    0  every requested client succeeded or was correctly skipped
    1  one or more clients failed (missing dataset, training error, ...)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# Make the project root importable when this script is run directly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Lock the artifact dir BEFORE importing ml_core (which reads ARTIFACT_DIR
# at import time and creates the dir).
os.environ.setdefault("ARTIFACT_DIR", os.path.join(_ROOT, "artifacts"))

from client_database import CLIENT_DB           # noqa: E402
from client_logic import get_logic              # noqa: E402
from ml_core import train_model, artifacts_exist  # noqa: E402


def train_one(ck: str, info: dict, force: bool = False) -> bool:
    """Train one client. Returns True on success or correct-skip, False on error."""
    if not info.get("has_embeddings"):
        print(f"[skip] {ck}: multiplier-only (no embeddings)")
        return True

    ds = info.get("dataset")
    if not ds or not os.path.exists(ds):
        print(f"[fail] {ck}: dataset missing ({ds})")
        return False

    L = get_logic(ck)

    if not force and artifacts_exist(ck, L.encoder_columns):
        print(f"[skip] {ck}: artifacts already present (use --force to retrain)")
        return True

    try:
        _, rmse = train_model(ck, ds, L, verbose=0)
        print(f"[ok]   {ck}: RMSE={rmse:.4f}")
        return True
    except Exception as e:
        print(f"[fail] {ck}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client", default="all",
                        help="Client key (e.g. 'tekion') or 'all'. Default: all.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if artifacts already exist.")
    args = parser.parse_args()

    if args.client == "all":
        targets = dict(CLIENT_DB)
    else:
        if args.client not in CLIENT_DB:
            print(f"[fail] unknown client '{args.client}'. "
                  f"Known: {sorted(CLIENT_DB)}")
            return 1
        targets = {args.client: CLIENT_DB[args.client]}

    print(f"Training {len(targets)} client(s) → "
          f"{os.environ['ARTIFACT_DIR']}")
    print()

    ok = True
    for ck, info in targets.items():
        if not train_one(ck, info, force=args.force):
            ok = False

    print()
    print("done." if ok else "one or more clients failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
