"""Shared helpers: load the participant-level dataset (data/participant_indices.csv) in the layout used by the analysis scripts."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "participant_indices.csv"
SEGMENTS = {"Previous": "baseline", "1": "LVLA", "2": "HVLA", "3": "LVHA", "4": "HVHA"}  # internal key -> column suffix
GROUP_CODE = {"A": "E", "B": "G", "C": "I", "D": "K"}  # internal group codes used in the original recordings


def load_indices(path=DATA):
    """Return one row per participant with columns label, order, sex, valid_<k>, val_<k> (FAA), aro_<k> (beta/(alpha+theta)),
    where <k> is Previous (baseline) or 1-4 (LVLA, HVLA, LVHA, HVHA)."""
    d = pd.read_csv(path, dtype={"presentation_order": str})
    out = pd.DataFrame({"participant_id": d.participant_id, "label": d.group.map(GROUP_CODE), "order": d.presentation_order,
                        "sex": d.sex, "text_cells_decoded": d.text_cells_decoded})
    for k, c in SEGMENTS.items():
        out[f"valid_{k}"] = d[f"n_valid_{c}"]
        out[f"val_{k}"] = d[f"faa_{c}"]
        out[f"aro_{k}"] = d[f"arousal_{c}"]
    return out
