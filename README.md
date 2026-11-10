# Human-response validation of AI-generated interior representations: EEG data and analysis code

Code and participant-level data for the manuscript *Human-Response Validation of AI-Generated Interior Representations: An
Exploratory EEG Study of Affective Correspondence* (revised version, R1).

Sixteen AI-generated residential interior images (four target valence–arousal quadrants × four prompt-to-image groups) were shown
to 96 participants (24 per group, 12 female and 12 male) while EEG was recorded with a Muse 2 headband (TP9, AF7, AF8, TP10). Each
participant saw the four images of one group for 15 s each after a 15-s eyes-closed baseline, in the order 1–2–3–4 or 4–3–2–1.

| Group | Image generator | Prompt source |
|---|---|---|
| A | MidJourney | ChatGPT |
| B | MidJourney | DeepSeek |
| C | Kling AI | ChatGPT |
| D | Kling AI | DeepSeek |

Target categories: LVLA = low valence–low arousal (image 1), HVLA = high valence–low arousal (2), LVHA = low valence–high arousal
(3), HVHA = high valence–high arousal (4).

## Repository contents

```
data/
  participant_indices.csv   one row per participant: group, sex, presentation order, per-segment EEG indices
  electrode_dropout.csv     channel-segments with loss of electrode contact (index values set to missing)
  band_summary.csv          band-level theta/alpha/beta power by segment and site (Supplementary Table S1)
  manifest_template.csv     layout of the manifest required to recompute the indices from raw exports
  data_dictionary.md        description of every column
scripts/
  common.py                          loads data/participant_indices.csv
  01_compute_indices_from_raw.py     raw Mind Monitor exports -> data/ (requires the non-public raw files)
  02_analysis.py                     Table 3, within-participant contrasts, Figures 2-4, Supplementary Figure S1
  03_quadrant_analysis.py            intended-quadrant correspondence vs. chance (Table 2)
  04_sensitivity_analysis.py         smallest effects detectable with 80% power (Section 2.12)
results/                             outputs of scripts 02-04 as used in the manuscript
requirements.txt
```

## Reproducing the analyses

```bash
pip install -r requirements.txt
python scripts/02_analysis.py
python scripts/03_quadrant_analysis.py
python scripts/04_sensitivity_analysis.py
```

Scripts 02 and 03 read `data/participant_indices.csv`, script 04 reads their outputs, and all write to `results/`:

| Output | Manuscript |
|---|---|
| `results/results.json`, `results/table3_platform_comparisons.csv` | Table 3; Section 3.1 (within-participant tests, category means); Section 3.2 (squared distance) |
| `results/quadrant_correspondence.json` | Table 2; Section 3.2 |
| `results/sensitivity.json` | Section 2.12 (sensitivity analysis); Sections 4 and 5 |
| `results/fig2_valence_by_platform.png`, `results/fig3_arousal_by_platform.png`, `results/fig4_squared_distance.png` | Figures 2–4 |
| `results/figS1_participant_distributions.png`, `data/band_summary.csv` | Supplementary Figure S1, Supplementary Table S1 |

The quadrant analysis uses a fixed random seed (10,000 within-participant permutations), so its output is reproducible exactly.

## Processing summary

All indices were computed from the absolute band powers exported by the Mind Monitor application (log10 values, about two samples
per second); details are in Sections 2.8–2.10 of the manuscript and in the docstring of `scripts/01_compute_indices_from_raw.py`.

* **Valence-related index:** frontal alpha asymmetry, ln(alpha AF8) − ln(alpha AF7), on linear power (10^x).
* **Arousal-related index:** beta / (alpha + theta), computed per channel on linear power and averaged over the four channels.
* **Change scores:** ΔValence = FAA(image) − FAA(baseline); ΔArousal = [A(image) − A(baseline)] / A(baseline). No truncation.
* **Missing data:** segments with fewer than 15 valid samples are treated as missing; consecutive duplicate samples are removed.
* **Text-stored cells:** 33,354 band-power cells had been stored as text with dot-separated digit groups during transcription.
  They correspond to absolute values ≥ 1 and are decoded by placing the decimal point after the first digit.
* **Electrode drop-out:** when all five bands of a channel fall below −1 within a segment, index values that depend on that channel
  in that segment are set to missing (`data/electrode_dropout.csv`; 4 valence-related and 11 arousal-related segment values in six
  recordings).
* **Verification against the study records:** the group assignment and presentation order of every recording were checked against
  the study records before analysis; recordings whose file labels were corrected are flagged in `participant_indices.csv`
  (`group_reassigned`, `order_from_study_records`), and recordings whose segment labels followed serial position were mapped to
  categories (`labels_mapped_from_position`).
* No filtering, ICA, or further artifact rejection was applied beyond the processing performed by the Mind Monitor application.

## Data availability and privacy

`data/` contains only derived, de-identified values. Participant IDs (A01–D24: group letter; 01–12 female, 13–24 male) are
arbitrary and cannot be linked to names. The raw Mind Monitor exports are not included because their file names and contents
identify participants; they are available from the corresponding author on reasonable request, subject to ethical approval. With
access to the raw files, `scripts/01_compute_indices_from_raw.py` reproduces `data/` exactly from a manifest in the format of
`data/manifest_template.csv`.

Self-Assessment Manikin ratings were collected after the EEG recording but were not analysed (manuscript Section 2.6) and are not
included.

## Citation

Please cite the article once published. *(Add the reference and DOI here.)*

## License

Code (`scripts/`): MIT License (see `LICENSE`). Data and results (`data/`, `results/`): Creative Commons Attribution 4.0
International (CC BY 4.0; see `data/LICENSE.md`).
