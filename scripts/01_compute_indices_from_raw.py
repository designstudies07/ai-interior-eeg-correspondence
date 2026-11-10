"""Compute the participant-level EEG indices from the labelled Mind Monitor exports.

Raw exports are not part of the public repository because their file names and contents identify participants. With access to
the raw files, run:
    python scripts/01_compute_indices_from_raw.py --raw-dir <folder> --manifest <manifest.csv>
The manifest has one row per recording: participant_id, file (path relative to --raw-dir), group (A-D), sex (F/M),
presentation_order (1234 or 4321, taken from the file name or, where the file name was wrong, from the study records),
group_reassigned, order_from_study_records (TRUE/FALSE, documentation only).

Processing (Sections 2.8-2.10 of the manuscript):
  * Band powers (Delta, Theta, Alpha, Beta, Gamma at TP9, AF7, AF8, TP10) are log10 values. Cells stored as text with
    dot-separated digit groups (e.g. '1.436.883.313.401.880') correspond to absolute values >= 1 and are decoded by placing the
    decimal point after the first digit.
  * Segments are taken from the 'Elements' column: Previous = eyes-closed baseline, 1-4 = LVLA, HVLA, LVHA, HVHA. If a
    recording with presentation order 4321 is labelled 1-2-3-4 in time, the labels are serial positions and are mapped to
    categories (1->HVHA, 2->LVHA, 3->HVLA, 4->LVLA).
  * Consecutive duplicate rows (written when no new packet arrives) are removed; n_valid = rows without missing band values.
  * Indices on linear power (10**x): FAA = ln(mean alpha AF8) - ln(mean alpha AF7); arousal = mean over the four channels of
    mean beta / (mean alpha + mean theta). Segments with < 3 valid rows get no index; the analyses use >= 15.
  * Electrode drop-out: if all five bands of a channel reach values <= -1 within a segment, index values that depend on that
    channel in that segment are set to missing (FAA: AF7/AF8; arousal: any channel).
Outputs: data/participant_indices.csv, data/electrode_dropout.csv, data/band_summary.csv
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT

SEG = ["Previous", "1", "2", "3", "4"]
NAME = {"Previous": "baseline", "1": "LVLA", "2": "HVLA", "3": "LVHA", "4": "HVHA"}
CH = ["TP9", "AF7", "AF8", "TP10"]
BANDS5 = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
FLIP = {"1": "4", "2": "3", "3": "2", "4": "1"}


def read(path):
    for sep in (";", ","):
        for enc in ("ISO-8859-1", "utf-8", "cp1254"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    return df
            except Exception:
                pass
    raise IOError(path)


def decode(v):
    """numeric value, or text cell with dot-separated digit groups -> decimal point after the first digit"""
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip()
    try:
        return float(t)
    except ValueError:
        pass
    sign = -1.0 if t.startswith("-") else 1.0
    p = t.lstrip("+-").split(".")
    if len(p) < 2 or not all(x.isdigit() for x in p):
        return math.nan
    d = "".join(p)
    return sign * float(d[0] + "." + d[1:])


def process(path, order):
    df = read(path)
    el = df["Elements"].astype(str).str.strip()
    bands = [c for c in df.columns if re.fullmatch(r"(Delta|Theta|Alpha|Beta|Gamma)_(TP9|AF7|AF8|TP10)", str(c))]
    n_text = 0
    for c in bands:
        if not pd.api.types.is_numeric_dtype(df[c]):
            as_num = pd.to_numeric(df[c], errors="coerce")
            n_text += int((as_num.isna() & df[c].notna()).sum())
            df[c] = df[c].map(decode).astype(float)
        else:
            df[c] = df[c].astype(float)
    first = next(x for x in el if x in SEG[1:])
    if order == "4321" and first == "1":  # labels are serial positions
        el = el.map(FLIP).fillna(el)
        mapped = True
    else:
        mapped = False
    keep = ~df[bands].eq(df[bands].shift()).all(axis=1)
    rec = {"text_cells_decoded": n_text, "labels_mapped_from_position": mapped}
    drop = []
    for s in SEG:
        m = el.eq(s)
        dropped = [ch for ch in CH if all((df.loc[m, f"{b}_{ch}"] <= -1).any() for b in BANDS5)]
        drop += [(NAME[s], ch) for ch in dropped]
        seg = df.loc[m & keep, bands].dropna(how="any")
        rec[f"n_valid_{NAME[s]}"] = len(seg)
        if len(seg) < 3:
            rec[f"faa_{NAME[s]}"] = rec[f"arousal_{NAME[s]}"] = np.nan
            continue
        P = 10.0 ** seg
        rec[f"faa_{NAME[s]}"] = np.nan if {"AF7", "AF8"} & set(dropped) else math.log(P["Alpha_AF8"].mean()) - math.log(P["Alpha_AF7"].mean())
        rec[f"arousal_{NAME[s]}"] = np.nan if dropped else float(np.mean([P[f"Beta_{c}"].mean() / (P[f"Alpha_{c}"].mean() + P[f"Theta_{c}"].mean()) for c in CH]))
    # band-level descriptives (Supplementary Table S1): theta/alpha/beta, frontal vs temporal, drop-out channels excluded
    b3 = [c for c in bands if c.split("_")[0] in ("Theta", "Alpha", "Beta")]
    d3 = df[b3].copy()
    keep3 = ~d3.eq(d3.shift()).all(axis=1)
    for seg_name, ch in drop:
        seg_key = next(k for k, v in NAME.items() if v == seg_name)
        d3.loc[el.eq(seg_key), [c for c in b3 if c.endswith("_" + ch)]] = np.nan
    bandrows = []
    for s in SEG:
        seg = d3.loc[el.eq(s) & keep3, b3]
        seg = seg.dropna(subset=[c for c in b3 if seg[c].notna().any()], how="any")
        if len(seg) < 15:
            continue
        for band in ("Theta", "Alpha", "Beta"):
            for site, chs in (("Frontal (AF7/AF8)", ("AF7", "AF8")), ("Temporal (TP9/TP10)", ("TP9", "TP10"))):
                mm = seg[[f"{band}_{c}" for c in chs]].mean()
                if mm.notna().any():
                    bandrows.append({"seg": {"Previous": "Baseline"}.get(s, NAME[s]), "band": band, "site": site, "v": mm.mean()})
    return rec, drop, bandrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    a = ap.parse_args()
    man = pd.read_csv(a.manifest, dtype={"presentation_order": str})
    rows, drops, bands = [], [], []
    for r in man.itertuples():
        rec, drop, br = process(Path(a.raw_dir) / r.file, r.presentation_order)
        rows.append({"participant_id": r.participant_id, "group": r.group,
                     "platform": {"A": "MidJourney", "B": "MidJourney", "C": "Kling AI", "D": "Kling AI"}[r.group],
                     "prompt_source": {"A": "ChatGPT", "B": "DeepSeek", "C": "ChatGPT", "D": "DeepSeek"}[r.group],
                     "sex": r.sex, "presentation_order": r.presentation_order, "group_reassigned": bool(r.group_reassigned),
                     "order_from_study_records": bool(r.order_from_study_records), **rec})
        drops += [{"participant_id": r.participant_id, "segment": s, "channel": c} for s, c in drop]
        bands += br
    out = Path(a.out_dir)
    cols = ["participant_id", "group", "platform", "prompt_source", "sex", "presentation_order", "group_reassigned",
            "order_from_study_records", "labels_mapped_from_position", "text_cells_decoded"]
    cols += [f"{p}_{NAME[s]}" for s in SEG for p in ("n_valid", "faa", "arousal")]
    D = pd.DataFrame(rows)[cols].sort_values("participant_id")
    for c in D.columns:
        if c.startswith(("faa_", "arousal_")):
            D[c] = D[c].round(6)
    D.to_csv(out / "participant_indices.csv", index=False)
    pd.DataFrame(drops, columns=["participant_id", "segment", "channel"]).sort_values(["participant_id", "segment"]).to_csv(out / "electrode_dropout.csv", index=False)
    P = pd.DataFrame(bands)
    P.groupby(["band", "site", "seg"])["v"].agg(["mean", "std", "count"]).reset_index().to_csv(out / "band_summary.csv", index=False)
    print(f"{len(D)} recordings; drop-out channel-segments: {len(drops)}; text cells decoded: {int(D.text_cells_decoded.sum())}")


if __name__ == "__main__":
    main()
