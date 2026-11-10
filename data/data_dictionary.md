# Data dictionary

## participant_indices.csv (96 rows, one per participant)

| Column | Description |
|---|---|
| `participant_id` | Arbitrary ID A01–D24 (group letter; 01–12 female, 13–24 male) |
| `group` | Prompt-to-image group: A (MidJourney–ChatGPT), B (MidJourney–DeepSeek), C (Kling AI–ChatGPT), D (Kling AI–DeepSeek) |
| `platform` | Image generator: MidJourney (A, B) or Kling AI (C, D) |
| `prompt_source` | Language model that generated the prompts: ChatGPT (A, C) or DeepSeek (B, D) |
| `sex` | F or M |
| `presentation_order` | 1234 (LVLA, HVLA, LVHA, HVHA) or 4321 (reverse) |
| `group_reassigned` | TRUE if the recording had been filed under another group and was reassigned from the study records |
| `order_from_study_records` | TRUE if the presentation order in the file name was wrong and was taken from the study records |
| `labels_mapped_from_position` | TRUE if the in-file segment labels were serial positions and were mapped to categories |
| `text_cells_decoded` | Number of band-power cells in the recording that had been stored as text and were decoded |
| `n_valid_<segment>` | Number of valid samples in the segment after removing consecutive duplicates (analyses require ≥ 15) |
| `faa_<segment>` | Frontal alpha asymmetry, ln(alpha AF8) − ln(alpha AF7), on linear power; empty = not computable or electrode drop-out |
| `arousal_<segment>` | Mean over TP9, AF7, AF8, TP10 of beta / (alpha + theta) on linear power; empty = not computable or electrode drop-out |

`<segment>` is `baseline` (15-s eyes-closed rest), `LVLA`, `HVLA`, `LVHA`, or `HVHA` (15-s viewing of the image of that target
category). Change scores used in the analyses: ΔValence = faa_<category> − faa_baseline;
ΔArousal = (arousal_<category> − arousal_baseline) / arousal_baseline.

## electrode_dropout.csv

| Column | Description |
|---|---|
| `participant_id` | Participant |
| `segment` | Segment in which all five bands of the channel fell below −1 (log10) |
| `channel` | Affected channel (TP9, AF7, AF8, TP10) |

## band_summary.csv (Supplementary Table S1)

| Column | Description |
|---|---|
| `band` | Theta, Alpha, or Beta |
| `site` | Frontal (AF7/AF8) or Temporal (TP9/TP10) |
| `seg` | Baseline, LVLA, HVLA, LVHA, HVHA |
| `mean`, `std`, `count` | Mean, standard deviation, and number of participants of the per-participant mean log10 band power |

## manifest_template.csv

Format of the manifest used by `scripts/01_compute_indices_from_raw.py`: `participant_id`, `file` (path of the participant's
labelled Mind Monitor export relative to `--raw-dir`), `group`, `sex`, `presentation_order`, `group_reassigned`,
`order_from_study_records`.
