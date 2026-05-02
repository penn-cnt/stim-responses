# stim-responses

Code and de-identified datasets to reproduce the main figures and machine learning analyses from:

> **[Paper title]** — [Authors], [Journal], [Year]

---

## Repository structure

```
stim-responses/
├── code/
│   ├── final-figures_v1.ipynb   # Main figures notebook
│   ├── inter_ML.py              # Random forest SOZ classifier
│   └── config.py                # Path configuration (portable)
├── datasets/                    # De-identified analysis datasets
│   ├── analysis_data_inter.csv
│   ├── analysis_data_prepost.csv
│   ├── analysis_data_morphology.csv
│   ├── analysis_data_rebound.csv
│   ├── subject_metadata.csv
│   ├── stim_channels.csv
│   ├── permanova_morphology_results_40mm.csv
│   └── permanova_F_null_40mm.npy
├── tools/
│   ├── iEEG_helper_functions.py
│   └── heatmap_gen.py
└── requirements.txt
```

Patient identifiers have been replaced with anonymous subject IDs (`sub_001`, `sub_002`, …). No mapping between anonymous IDs and original identifiers is stored or provided.

---

## Setup

### Requirements

- Python 3.12
- pip

### Install dependencies

Create and activate a virtual environment, then install:

```bash
python3.12 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

All packages and their exact versions are listed in `requirements.txt`. No conda is required.

### Register the kernel (for notebook use)

```bash
python -m ipykernel install --user --name stim-responses --display-name "stim-responses"
```

---

## Reproducing the results

### Figures (notebook)

All main figures are produced by running `code/final-figures_v1.ipynb` from top to bottom.

```bash
cd code
jupyter lab final-figures_v1.ipynb
```

Select the `stim-responses` kernel, then **Run All Cells** (`Kernel → Restart Kernel and Run All Cells`).

The notebook is self-contained: every analysis section begins with a data loading cell that reads directly from `datasets/`. No intermediate steps are needed.

Section markers in the notebook:
| Section header | Figures produced |
|----------------|-----------------|
| `# Figure 1` | Inter-stimulation spike rate changes |
| `# Figure 2` | Spike rate changes by brain region |
| `# Figure 3 (now part of figure 2)` | MTL/MTLE subgroup comparisons |
| `# Figure 4` | Spike morphology and PERMANOVA |
| `# Figure 5` | Random forest SOZ classification (summary) |

### Random forest classifier

The full LOOCV random forest analysis (Figure 5) can also be run as a standalone script:

```bash
cd code
python inter_ML.py
```

This runs patient-level leave-one-out cross-validation for two models (full 3-feature model vs. baseline spike-rate-only model), prints a comparison table with bootstrap confidence intervals and DeLong's test, and saves results to `results/ML-results/`.

Expected runtime: 30–90 minutes depending on CPU (parallelized via `n_jobs=-1`).

---

## Datasets

| File | Description | Rows |
|------|-------------|------|
| `analysis_data_inter.csv` | Per-channel spike rates during and between stimulation trials, with 24hr baseline and electrode distances merged | ~233k |
| `analysis_data_prepost.csv` | Same structure for pre/during/post stimulation epochs | ~233k |
| `analysis_data_morphology.csv` | Individual spike morphology features per trial | ~161k |
| `analysis_data_rebound.csv` | Post-stimulation rebound spike rates in 10s bins | ~131k |
| `subject_metadata.csv` | Per-subject clinical labels: stimulation-induced seizure status, MTL/EC classification | 43 subjects |
| `stim_channels.csv` | Which electrode channels were stimulated per subject | ~3k |
| `permanova_morphology_results_40mm.csv` | PERMANOVA results for spike morphology analysis | — |
| `permanova_F_null_40mm.npy` | Permutation null distribution for PERMANOVA F-statistic | — |

All datasets use `subject_id` as the anonymous patient identifier.
