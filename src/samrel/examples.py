"""Rule-based choice of example images for the success/failure figure (PROTOCOL §13)."""

from __future__ import annotations

import pandas as pd

RULES = ("typical", "overconfident", "a0_worst", "a2_worst")
DATASETS = ("colondb", "etis")


def _pick(scores: pd.Series, taken: set[str], ascending: bool) -> str:
    """Best-scoring image not yet taken; ties go to the smaller image_id."""
    order = scores.reset_index().sort_values([scores.name, "image_id"], ascending=[ascending, True])
    for image_id in order["image_id"]:
        if image_id not in taken:
            return image_id
    raise ValueError("no image left to pick")


def select_examples(a0: pd.DataFrame, a2_seed0: pd.DataFrame,
                    datasets=DATASETS, rules=RULES) -> pd.DataFrame:
    """Rows ``dataset, rule, image_id`` in rule order. Inputs are per-image tables (PROTOCOL §8)."""
    out = []
    for ds in datasets:
        def prep(df):
            g = df[(df["dataset"] == ds) & ~df["excluded"].astype(bool)].set_index("image_id")
            return g
        g0, g2 = prep(a0), prep(a2_seed0)
        if set(g0.index) != set(g2.index):
            raise ValueError(f"{ds}: A0 and A2 cover different images")
        taken: set[str] = set()
        for rule in rules:
            if rule == "typical":
                s = (g0["dice"] - g0["dice"].median()).abs().rename("score")
                pick = _pick(s, taken, ascending=True)
            elif rule == "overconfident":
                s = (g2["conf_iou_pred"] - g2["iou"]).rename("score")
                pick = _pick(s, taken, ascending=False)
            elif rule == "a0_worst":
                pick = _pick(g0["dice"].rename("score"), taken, ascending=True)
            elif rule == "a2_worst":
                pick = _pick(g2["dice"].rename("score"), taken, ascending=True)
            else:
                raise ValueError(rule)
            taken.add(pick)
            out.append({"dataset": ds, "rule": rule, "image_id": pick})
    return pd.DataFrame(out)
