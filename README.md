# Negative semantic recurrence during spontaneous thought

Code for reproducing the **main analyses** and quantitative figures in *Negative semantic recurrence during spontaneous thought relates to automatic thought tendency and posterior parahippocampal connectivity organization*.

This repository covers MPNet-based mNSRL, signed PMC–MTL connectivity, participant-specific SVD, statistical associations, and standardized indirect association. Supplementary-only analyses and journal figure artwork are not included.

## Structure

```text
negative-semantic-recurrence/
├── config/
│   └── imaging_exclusions.txt
├── results/
│   └── .gitkeep
├── scripts/
│   ├── 00_prepare_inputs.py
│   ├── 01_extract_connectivity.py
│   ├── 02_semantic_recurrence.py
│   ├── 03_main_analysis.py
│   ├── 04_indirect_association.R
│   └── 05_plots.py
├── utils.py
└── README.md
```

## Input data

Download the publicly released [OpenNeuro ds006067 v2.0.0](https://openneuro.org/datasets/ds006067/versions/2.0.0) fMRIPrep derivatives and [OSF Think Aloud Behavioral Data](https://osf.io/a56rm/).

- **Behavioral data:** Extracted OSF directory (or original ZIP), including sentence-level emotional-intensity annotations, timestamped transcripts, and `ThinkAloudQuestionnaires_summary.csv`.
- **Participant metadata:** Original `participants.txt` (or `participants.tsv`) containing participant IDs, age, and sex when available.
- **fMRI:** Preprocessed BOLD images and confounds TSV files in the fMRIPrep derivatives directory.

`config/imaging_exclusions.txt` specifies the participants excluded by imaging quality control. Step 0 automatically selects participants not in that list who have complete ATQ-30, age, sex, and sentence-level annotation data; no separate manuscript-subject list is needed. Step 1 uses the selected participants generated in Step 0. For the released dataset and exclusion criteria described in the manuscript, the expected final analytic sample is **N = 43**. Review `results/tables/source_report.json` and the final N printed by Step 3 to verify the cohort. Source questionnaire scores and emotional-intensity annotations are not recomputed. No raw participant data are distributed in this repository.

## Installation

Use **Python 3.10+** and R (for Step 4). From the repository root:

```powershell
conda create -n nsr python=3.11 -y
conda activate nsr
python -m pip install numpy pandas openpyxl scipy statsmodels matplotlib sentence-transformers nibabel nilearn templateflow tqdm
```

Install the R mediation package once:

```r
install.packages("mediation")
```

The first execution may download the MPNet embedding model and TemplateFlow atlas resources.

## Run in order

Open **PowerShell in the repository root** and run each command, replacing the input paths with your local paths.

**Step 0 — Prepare behavioral inputs and select the analytic participants**

```powershell
python scripts/00_prepare_inputs.py --behavior "C:\path\to\beh_data" --subject-info "E:\path\to\participants.txt"
```

**Step 1 — Extract PMC–MTL connectivity**

```powershell
python scripts/01_extract_connectivity.py --fmriprep "E:\path\to\derivatives" --subjects-file ./results/tables/selected_subjects.txt
```

**Step 2 — Calculate negative semantic recurrence**

```powershell
python scripts/02_semantic_recurrence.py --sentences ./results/inputs/sentences.csv
```

**Step 3 — Main statistics and participant-specific SVD**

```powershell
python scripts/03_main_analysis.py --participants ./results/inputs/participants.csv
```

**Step 4 — Standardized indirect association (R)**

```powershell
Rscript scripts/04_indirect_association.R ./results/tables/analysis_dataset.csv ./results/tables 10000 42
```

**Step 5 — Generate main figures and CI table**

```powershell
python scripts/05_plots.py
```

## Outputs

All generated outputs are written to `results/`:

- `results/inputs/`: standardized participant and sentence tables.
- `results/tables/`: source QC, selected participant IDs, semantic recurrence, FC/SVD, statistical results, and mediation outputs.
- `results/figures/`: Figure 1–4 PNG/TIFF files, panel-source CSV files, and `ci_summary.csv`.

Sentence-order permutations use seed `20260822`. Participant bootstrap analyses and R mediation use seed `42` by default. All plotting helpers are included in `utils.py`.
