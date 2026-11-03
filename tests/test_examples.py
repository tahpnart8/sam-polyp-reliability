"""PROTOCOL §13 example selection on a hand-made table."""

from __future__ import annotations

import pandas as pd
import pytest

from samrel.examples import select_examples


def table(dice, conf):
    ids = [f"etis/{i}" for i in range(len(dice))]
    return pd.DataFrame({"dataset": "etis", "image_id": ids, "dice": dice, "conf_iou_pred": conf, "excluded": False})


def test_rules_and_no_duplicates():
    a0 = table([0.9, 0.5, 0.7, 0.8, 0.2], [0.9] * 5)
    a2 = table([0.95, 0.4, 0.9, 0.9, 0.85], [0.9, 0.95, 0.9, 0.9, 0.9])
    s = select_examples(a0, a2, datasets=("etis",)).set_index("rule")["image_id"]
    assert s["typical"] == "etis/2"          # median 0.7
    assert s["overconfident"] == "etis/1"    # 0.95 - 0.4
    assert s["a0_worst"] == "etis/4"
    assert s["a2_worst"] == "etis/3"          # 1, 4, 2 already taken; 3 is next (tie with 2 by id)
    assert len(set(s)) == 4


def test_tie_goes_to_smaller_id_and_excluded_skipped():
    a0 = table([0.5, 0.5, 0.5], [0.9] * 3)
    a2 = table([0.5, 0.5, 0.5], [0.9] * 3)
    a0.loc[0, "excluded"] = a2.loc[0, "excluded"] = True
    s = select_examples(a0, a2, datasets=("etis",), rules=("typical", "overconfident")).set_index("rule")["image_id"]
    assert s["typical"] == "etis/1" and s["overconfident"] == "etis/2"


def test_mismatched_images_rejected():
    with pytest.raises(ValueError):
        select_examples(table([0.5, 0.6], [0.9, 0.9]), table([0.5], [0.9]), datasets=("etis",))
