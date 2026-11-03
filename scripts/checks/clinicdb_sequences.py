"""Map PraNet CVC-ClinicDB train/test images to video sequences with the official frame table.

Source: CVC-ClinicDB page on Grand Challenge (https://polyp.grand-challenge.org/CVCClinicDB/),
"Table 1" image (cvc-clinicdbgt2.png), viewed 2026-11-02: 29 sequences, contiguous frame ranges.
The paper of the dataset (Bernal et al. 2015) and EndoScene give 31 sequences / 23 patients; the
two counts are not reconciled by any source found, and no frame-to-patient table is published.

Assumption, stated in the output: a PraNet file name ``<n>.png`` is original frame ``n``. Evidence:
the 550 train and 62 test names are integers whose union is exactly 1..612 (checked here).

    python scripts/checks/clinicdb_sequences.py --data-root data/raw --out-dir results/data_checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from samrel.config import load_config
from samrel.data import discover

# sequence -> (first frame, last frame), Grand Challenge Table 1
SEQUENCES = {
    1: (1, 25), 2: (26, 50), 3: (51, 67), 4: (68, 78), 5: (79, 103), 6: (104, 126), 7: (127, 151),
    8: (152, 177), 9: (178, 199), 10: (200, 205), 11: (206, 227), 12: (228, 252), 13: (253, 277),
    14: (278, 297), 15: (298, 317), 16: (318, 342), 17: (343, 363), 18: (364, 383), 19: (384, 408),
    20: (409, 428), 21: (429, 447), 22: (448, 466), 23: (467, 478), 24: (479, 503), 25: (504, 528),
    26: (529, 546), 27: (547, 571), 28: (572, 591), 29: (592, 612),
}


def sequence_of(frame: int) -> int:
    for s, (a, b) in SEQUENCES.items():
        if a <= frame <= b:
            return s
    raise ValueError(f"frame {frame} outside 1..612")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    covered = sorted(f for a, b in SEQUENCES.values() for f in range(a, b + 1))
    assert covered == list(range(1, 613)), "official table must cover 1..612 exactly once"
    recs = [r for r in discover(args.data_root, load_config("base")["data"]) if r.dataset in ("clinicdb_train", "clinicdb")]
    frames = {ds: {int(r.image_id.split("/", 1)[1]) for r in recs if r.dataset == ds and r.image_path}
              for ds in ("clinicdb_train", "clinicdb")}
    train, test = frames["clinicdb_train"], frames["clinicdb"]
    if train & test or (train | test) != set(range(1, 613)):
        raise SystemExit("STOP: PraNet ClinicDB names are not a partition of 1..612; the mapping does not apply")

    rows = []
    for s, (a, b) in SEQUENCES.items():
        fs = range(a, b + 1)
        rows.append({"sequence": s, "first_frame": a, "last_frame": b, "n_frames": b - a + 1,
                     "n_train": sum(f in train for f in fs), "n_test": sum(f in test for f in fs)})
    seq = pd.DataFrame(rows)
    seq["shared"] = (seq.n_train > 0) & (seq.n_test > 0)
    per_image = pd.DataFrame({"image_id": [f"clinicdb/{f}" for f in sorted(test)],
                              "frame": sorted(test), "sequence": [sequence_of(f) for f in sorted(test)]})
    per_image["sequence_in_train"] = per_image.sequence.map(seq.set_index("sequence").n_train > 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq.to_csv(args.out_dir / "clinicdb_sequences.csv", index=False)
    per_image.to_csv(args.out_dir / "clinicdb_test_sequence.csv", index=False)
    n_seq_test = int((seq.n_test > 0).sum())
    print(f"test images {len(test)} in {n_seq_test} of 29 sequences; sequences shared with train "
          f"{int(seq.shared.sum())}; test images whose sequence also has train frames "
          f"{int(per_image.sequence_in_train.sum())}/{len(test)}; sequences only in train "
          f"{seq.loc[seq.n_test == 0, 'sequence'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
