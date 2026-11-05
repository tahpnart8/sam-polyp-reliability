"""Independent recomputation of the paired comparisons from per-image files.

Uses numpy and scipy.stats.rankdata only, no samrel code: own seed derivation, vectorised metrics,
own consistency reading. Part 1 reproduces paired.csv, paired_consistency.csv and perturb/rq2_pairs.csv
with the official index matrices. Part 2 draws a fresh matrix (other seed label, B = 10,000) to see which
readings depend on the Monte Carlo draw; it changes nothing in the analysis.

Usage:
    python scripts/checks/paired_independent.py --metrics-dir results/metrics \
        --out-csv results/data_checks/paired_independent.csv --report results/logs/paired_independent_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT_SEED = 20261029
REGION = {"img": 0, "roi": 1, "bnd": 2}
METRICS = ("dice", "iou", "ece_img", "ece_roi", "ece_bnd", "aurc_iou_pred", "aurc_mean_inside",
           "spearman_iou_pred", "spearman_mean_inside")
SCORE = {"iou_pred": "conf_iou_pred", "mean_inside": "conf_mean_inside"}


def seed_of(*labels) -> int:
    return int.from_bytes(hashlib.sha256(repr(labels).encode()).digest()[:4], "big") % (2 ** 31)


def spearman(conf: np.ndarray, dice: np.ndarray) -> np.ndarray:
    """Pearson on average ranks, row by row; conf and dice are [B, n]."""
    a, b = rankdata(conf, axis=1), rankdata(dice, axis=1)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def aurc(conf: np.ndarray, dice: np.ndarray) -> np.ndarray:
    order = np.argsort(-conf, axis=1, kind="stable")
    risk = np.cumsum(1.0 - np.take_along_axis(dice, order, axis=1), axis=1) / np.arange(1, conf.shape[1] + 1)
    return risk.mean(1)


def ece(stats: np.ndarray, idx: np.ndarray, region: int) -> np.ndarray:
    n = stats.shape[0]
    weights = np.zeros((idx.shape[0], n))
    np.add.at(weights, (np.arange(idx.shape[0])[:, None], idx), 1.0)
    s = (weights @ stats[:, region].reshape(n, -1)).reshape(idx.shape[0], -1, 3)
    return np.abs(s[:, :, 2] - s[:, :, 1]).sum(1) / s[:, :, 0].sum(1)


def values(df: pd.DataFrame, stats: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    dice, iou = df["dice"].to_numpy(float), df["iou"].to_numpy(float)
    out = {"dice": dice[idx].mean(1), "iou": iou[idx].mean(1)}
    for name, k in REGION.items():
        out[f"ece_{name}"] = ece(stats, idx, k)
    for name, col in SCORE.items():
        c = df[col].to_numpy(float)
        out[f"aurc_{name}"] = aurc(c[idx], dice[idx])
        out[f"spearman_{name}"] = spearman(c[idx], dice[idx])
    return out


def load(per_image: Path, bins: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(per_image)
    df = df[~df["excluded"].astype(bool)].sort_values("image_id").reset_index(drop=True)
    with np.load(bins) as z:
        pos = {i: k for k, i in enumerate(z["image_ids"].tolist())}
        return df, z["stats"][[pos[i] for i in df["image_id"]]]


class Runs:
    """Values of one group of runs, sharing one index matrix per dataset."""

    def __init__(self, tables: dict, root: tuple, B: int):
        self.tables, self.root, self.B, self.cache, self.idx = tables, root, B, {}, {}

    def get(self, key, ds):
        if (key, ds) not in self.cache:
            df, st = self.tables[key]
            m = (df["dataset"] == ds).to_numpy()
            sub, sub_st = df[m].reset_index(drop=True), st[m]
            n = len(sub)
            if ds not in self.idx:
                self.idx[ds] = np.random.default_rng(seed_of(*self.root, ds)).integers(0, n, size=(self.B, n))
            self.cache[(key, ds)] = (values(sub, sub_st, np.arange(n)[None, :]), values(sub, sub_st, self.idx[ds]))
        return self.cache[(key, ds)]


def diff_ci(runs: Runs, a, b, ds, metric):
    fa, sa = runs.get(a, ds)
    fb, sb = runs.get(b, ds)
    lo, hi = np.quantile(sa[metric] - sb[metric], [0.025, 0.975])
    return float(fa[metric][0] - fb[metric][0]), float(lo), float(hi)


def sign(lo, hi):
    return np.where(lo > 0, 1, np.where(hi < 0, -1, 0))


def read_family(frame: pd.DataFrame, col: str) -> dict:
    out = {}
    for k, g in frame.groupby(["source", "family", "level", "dataset", "metric"], sort=True):
        s = set(g[col].astype(int))
        out[k] = next(iter(s)) if len(s) == 1 and 0 not in s and len(g) == 3 else 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--check-B", type=int, default=10000)
    args = ap.parse_args(argv)
    mdir, pdir = args.metrics_dir, args.metrics_dir / "perturb"

    runs = pd.read_csv(mdir / "runs.csv", dtype=str)
    tables = {f"{r.config}_s{r.seed}": load(mdir / "per_image" / f"{r.run_id}.csv", mdir / "ece_bins" / f"{r.run_id}.npz")
              for r in runs.itertuples()}
    bad_formula = 0
    for df, _ in tables.values():
        d = df["dice"].to_numpy(float)
        bad_formula += int((np.abs(df["iou"].to_numpy(float) - d / (2 - d)) > 1e-9).sum())
        bad_formula += int((df["failed"].astype(bool).to_numpy() != (d < 0.5)).sum())

    official = pd.read_csv(mdir / "paired.csv")
    official = official[official["metric"].isin(METRICS)]
    main_runs = Runs(tables, (ROOT_SEED, "bootstrap"), 1000)
    rows = []
    for o in official.itertuples():
        d, lo, hi = diff_ci(main_runs, o.config_a, o.config_b, o.dataset, o.metric)
        rows.append({"source": "paired.csv", "a": o.config_a, "b": o.config_b, "level": "P0", "dataset": o.dataset,
                     "metric": o.metric, "diff_official": o.diff, "ci_low_official": o.ci_low,
                     "ci_high_official": o.ci_high, "diff": d, "ci_low": lo, "ci_high": hi})

    pruns = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(pdir.glob("runs_*.csv"))], ignore_index=True)
    ptables = {(r.config, r.seed, r.level): load(pdir / "per_image" / f"{r.run_id}.csv", pdir / "ece_bins" / f"{r.run_id}.npz")
               for r in pruns.itertuples()}
    prq2 = pd.read_csv(pdir / "rq2_pairs.csv", dtype={"seed_a": str, "seed_b": str})
    prq2 = prq2[prq2["metric"].isin(METRICS)]
    pr = Runs(ptables, (ROOT_SEED, "bootstrap"), 1000)
    for o in prq2.itertuples():
        ca, cb = o.family.split("_vs_")
        d, lo, hi = diff_ci(pr, (ca, o.seed_a, o.level), (cb, o.seed_b, o.level), o.dataset, o.metric)
        rows.append({"source": "rq2_pairs.csv", "a": f"{ca}_s{o.seed_a}", "b": f"{cb}_s{o.seed_b}", "level": o.level,
                     "dataset": o.dataset, "metric": o.metric, "diff_official": o.diff, "ci_low_official": o.ci_low,
                     "ci_high_official": o.ci_high, "diff": d, "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(rows)

    check = Runs(tables, (ROOT_SEED, "bootstrap_check"), args.check_B)
    pcheck = Runs(ptables, (ROOT_SEED, "bootstrap_check"), args.check_B)
    lows, highs = [], []
    for r in res.itertuples():
        if r.dataset not in ("colondb", "etis"):
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        if r.source == "paired.csv":
            _, lo, hi = diff_ci(check, r.a, r.b, r.dataset, r.metric)
        else:
            ca, sa = r.a.split("_s")
            cb, sb = r.b.split("_s")
            _, lo, hi = diff_ci(pcheck, (ca, sa, r.level), (cb, sb, r.level), r.dataset, r.metric)
        lows.append(lo)
        highs.append(hi)
    res["ci_low_check"], res["ci_high_check"] = lows, highs
    res["sign_official"] = sign(res["ci_low_official"], res["ci_high_official"])
    res["sign"] = sign(res["ci_low"], res["ci_high"])
    res["sign_check"] = sign(res["ci_low_check"], res["ci_high_check"])
    res["family"] = res["a"].str.split("_").str[0] + "_vs_" + res["b"].str.split("_").str[0]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)

    ext = res[res["dataset"].isin(["colondb", "etis"])]
    off, chk = read_family(ext, "sign_official"), read_family(ext, "sign_check")
    flips = [(k, off[k], chk[k]) for k in off if off[k] != chk[k]]
    seed_flips = int((ext["sign_official"] != ext["sign_check"]).sum())
    consist = pd.read_csv(mdir / "paired_consistency.csv")
    consist = consist[consist["metric"].isin(METRICS)]
    n_mismatch = 0
    for c in consist.itertuples():
        g = res[(res["source"] == "paired.csv") & (res["family"] == c.family) & (res["dataset"] == c.dataset)
                & (res["metric"] == c.metric)]
        mine = len(g) == 3 and len(set(g["sign"])) == 1 and 0 not in set(g["sign"])
        n_mismatch += int(bool(c.consistent) != mine)

    def gap(a, b):
        return float((res[a] - res[b]).abs().max())

    lines = ["# Independent recomputation of paired comparisons", "",
             f"- Rows compared: {len(res)} ({(res['source'] == 'paired.csv').sum()} from paired.csv, "
             f"{(res['source'] == 'rq2_pairs.csv').sum()} from perturb/rq2_pairs.csv).",
             f"- Metrics: {', '.join(METRICS)}.",
             f"- IoU = Dice/(2-Dice) and failed = Dice < 0.5 on every per-image row of the 8 runs: {bad_formula} violations.",
             f"- Largest absolute gap, point difference: {gap('diff', 'diff_official'):.2e}.",
             f"- Largest absolute gap, CI low: {gap('ci_low', 'ci_low_official'):.2e}; CI high: {gap('ci_high', 'ci_high_official'):.2e}.",
             f"- Rows whose sign class differs from the official one: {int((res['sign'] != res['sign_official']).sum())}.",
             f"- paired_consistency.csv `consistent` rows that differ: {n_mismatch} of {len(consist)}.",
             "", f"## Fresh index matrix, B = {args.check_B} (external sets)", "",
             f"- Seed-level rows compared: {len(ext)}; sign class changed: {seed_flips}.",
             f"- Family readings compared: {len(off)}; changed: {len(flips)}.", ""]
    lines += [f"- {k[0]} {k[1]} {k[2]} {k[3]} {k[4]}: official {v0:+d}, fresh matrix {v1:+d}" for k, v0, v1 in flips]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
