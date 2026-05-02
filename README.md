# stim-responses

Code and de-identified datasets to reproduce the main figures and machine learning analyses from:

> **Interictal spike rates track acute changes in cortical excitability during stimulation in drug-resistant epilepsy**
> Carlos A. Aguila, Zican Zhuo, Sarah B. Lavelle, William K.S. Ojemann, Juri Kim, Katherine G Walsh, Sasan Sedighi Mournani, Alfredo Lucas, Nishant Sinha, Odile Feys, Kathryn A. Davis, Brian Litt, Erin C. Conrad
>
> Correspondence: Carlos A. Aguila — aguilac@seas.upenn.edu

Patient identifiers have been replaced with anonymous subject IDs (`sub_001`, `sub_002`, …). No mapping between anonymous IDs and original identifiers is stored or provided.

---

## Repository structure

```
stim-responses/
├── code/
│   ├── final-figures_v1.ipynb   # Main figures (Figures 2–5)
│   ├── inter_ML.py              # Random forest SOZ classifier (Figure 5)
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

---

## Setup

### Requirements

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Create the environment

```bash
conda create -n stim-responses python=3.12
conda activate stim-responses
conda install pip
pip install -r requirements.txt
```

### Register the kernel (for notebook use)

```bash
python -m ipykernel install --user --name stim-responses --display-name "stim-responses"
```

---

## Reproducing the results

### Figures 2–4 (notebook)

All main figures are produced by running `code/final-figures_v1.ipynb` from top to bottom.

```bash
cd code
jupyter lab final-figures_v1.ipynb
```

Select the `stim-responses` kernel, then **Kernel → Restart Kernel and Run All Cells**.

Each figure section begins with a data loading cell that reads directly from `datasets/` — no intermediate preprocessing steps are needed.

| Notebook section | Figure | Description |
|------------------|--------|-------------|
| `# Figure 1` | — | Methods figure (not reproduced here) |
| `# Figure 2` | Fig. 2 | LFS increases IZ spike rate in a distance-dependent manner; spike rates decay toward zero beyond 40 mm and recover ~30 s post-stimulation offset |
| `# Figure 3 (now part of figure 2)` | Fig. 3 | MTL stimulation produces the largest spike rate increase across brain regions; effect is present in both MTLE and non-MTLE patients and independent of stimulation-induced seizures |
| `# Figure 4` | Fig. 4 | Stimulation transiently alters spike morphology (PERMANOVA p < 0.001), but the condition effect accounts for only 0.36% of total variance; during-stimulation spikes are wider and less sharp |
| `# Figure 5` | Fig. 5 | Summary of SOZ classifier results (full plots produced by `inter_ML.py`) |

### Figure 5 — Random forest SOZ classifier

```bash
cd code
python inter_ML.py
```

Runs patient-level leave-one-out cross-validation (LOOCV) comparing a full 3-feature random forest model (baseline spike rate + two stimulation-evoked features) against a baseline-only model. Stimulation features did not improve SOZ localization beyond baseline spike rate alone (DeLong's p = 0.81; full model AUC = 0.787, baseline AUC = 0.747). Results and figures are saved to `results/ML-results/`.

Expected runtime: 30–90 minutes depending on CPU (parallelized via `n_jobs=-1`).

---

## Datasets

| File | Description | Rows |
|------|-------------|------|
| `analysis_data_inter.csv` | Per-channel spike rates during and between stimulation trials, with 24 hr baseline and electrode distances merged | ~233k |
| `analysis_data_prepost.csv` | Same structure for pre / during / post stimulation epochs | ~233k |
| `analysis_data_morphology.csv` | Individual spike morphology features per trial, with electrode distances merged | ~161k |
| `analysis_data_rebound.csv` | Post-stimulation rebound spike rates in 10 s bins | ~131k |
| `subject_metadata.csv` | Per-subject clinical labels: stimulation-induced seizure status, MTL/EC classification | 43 subjects |
| `stim_channels.csv` | Which electrode channels were stimulated per subject | ~3k |
| `permanova_morphology_results_40mm.csv` | PERMANOVA results for spike morphology analysis | — |
| `permanova_F_null_40mm.npy` | Permutation null distribution for PERMANOVA F-statistic | — |

All datasets use `subject_id` as the anonymous patient identifier.
