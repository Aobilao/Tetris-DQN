import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from tetris_rl.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"^const MODEL = \{.*\};$", re.MULTILINE)


def extract(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["online_state_dict"]
    norm = ckpt["normalizer"]

    n_features = sd["fc1.weight"].shape[1]
    d_ff = sd["fc1.weight"].shape[0]
    flat = lambda t: [round(float(v), 6) for v in t.flatten().tolist()]

    return {
        "features": list(FEATURE_NAMES[:n_features]),
        "mean": [round(float(v), 6) for v in norm.mean],
        "std": [round(float(v), 6) for v in np.sqrt(norm.var)],
        "w1": flat(sd["fc1.weight"]),
        "b1": flat(sd["fc1.bias"]),
        "w2": flat(sd["fc2.weight"]),
        "b2": flat(sd["fc2.bias"]),
        "w3": flat(sd["fc3.weight"]),
        "b3": flat(sd["fc3.bias"]),
        "d_ff": d_ff,
        "n_features": n_features,
        "step": int(ckpt.get("step", 0)),
        "best_reward": round(float(ckpt.get("best_reward", float("nan"))), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--page", type=Path, default=ROOT / "web" / "index.html")
    args = ap.parse_args()

    model = extract(args.checkpoint)
    line = "const MODEL = " + json.dumps(model, separators=(",", ":")) + ";"

    html = args.page.read_text()
    if not PATTERN.search(html):
        raise SystemExit(f"no `const MODEL = ...;` line found in {args.page}")
    args.page.write_text(PATTERN.sub(lambda _: line, html, count=1))

    print(
        f"{args.page.relative_to(ROOT)} <- {args.checkpoint.name} "
        f"(step {model['step']:,}, best_reward {model['best_reward']}, "
        f"{model['n_features']} features, d_ff {model['d_ff']})"
    )


if __name__ == "__main__":
    main()
