# CS 599: Lab 2 — Catastrophic Forgetting on Permuted-MNIST

This package contains **fully working TensorFlow 2** code to reproduce the assignment results:
- Trains an **MLP** with depths **2 / 3 / 4** (256 units each)
- Dataset: **Permuted MNIST** with **10 tasks**
- Schedule: **50 epochs** for Task 1 + **20 epochs** for Tasks 2–10 (**230 total**)
- Experiments: **loss (NLL, L1, L2, L1+L2)** via kernel regularization, **optimizers** (SGD, Adam, RMSprop), **dropout** (0–0.5)
- Metrics: **R matrix**, **ACC**, **BWT**, plus **TBWT** and **CBWT** (bonus)
- Artifacts saved per run: `R.npy`, `R.csv`, `metrics.json`, `last_row.png`, `summary.txt`

> Designed to meet the requirements in the lab brief. See the citations in your report for the original definitions of ACC and BWT.


## Quick Start

```bash
# 1) (optional) Create virtual env and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Single run: 3-layer MLP, Adam, NLL, dropout 0.3
python3 src/forgetting_mlp.py --depth 3 --loss nll --optimizer adam --dropout 0.3

# 3) Sweep all depths with SGD and dropout 0.5
python3 src/forgetting_mlp.py --depth all --loss nll --optimizer sgd --dropout 0.5

# 4) Full sweep (careful: long)
bash run_all.sh
```

Results are saved under `outputs/<tag>/`. Open `summary.txt` and `metrics.json` for each run. The **R.csv** file is ready to import into LaTeX tables/plots.


## Command-Line Options

- `--depth {2,3,4,all}` — number of hidden layers (each 256 units). `all` runs 2, 3, and 4.
- `--loss {nll,l1,l2,l1+l2}` — NLL with optional L1/L2/L1+L2 kernel regularization.
- `--optimizer {sgd,adam,rmsprop}` — optimizer choice.
- `--dropout [0..0.5]` — dropout prob (assignment requires ≤ 0.5).
- `--tasks` — number of tasks (default 10).
- `--epochs_first` — epochs for Task 1 (default 50).
- `--epochs_rest` — epochs for remaining tasks (default 20).
- `--seed` — global seed (default 5695). **Use your unique seed**.
- `--outdir` — where to save outputs.
- `--reg_scale` — regularization scale for L1/L2 (default 1e-4).
- `--verbosity` — Keras fit verbosity.


## What gets computed

- **R matrix** (T×T): `R[t, i]` = accuracy on Task *i* after finishing training on Task *t* (0‑indexed).
- **ACC** (Average Accuracy): mean of last row of `R`.
- **BWT** (Backward Transfer): average of `R[T-1, i] − R[i, i]` for `i < T-1`.
- **TBWT** and **CBWT**: aggregated variants as described in literature (bonus).

The **last-row plot** `last_row.png` shows accuracy for each task after finishing all tasks.


## LaTeX (NeurIPS) Report Template

A starter `report_template.tex` is included. Replace the document class with the official **NeurIPS** template used by your course before submission. The template shows where to paste key numbers (ACC, BWT, etc.).


## Notes & Tips

- **Interpretation of L1/L2**: The brief lists *losses* (NLL, L1, L2, L1+L2). In standard practice, L1/L2 are applied as **kernel regularization** terms while the supervised loss is still NLL. This code follows that convention.
- **Reproducibility**: `--seed` fixes the permutations across tasks and TF initializations.
- **Runtime**: The full sweep is compute‑heavy. Start with a couple of runs to sanity‑check.


## File Tree

```
.
├── README.md
├── requirements.txt
├── run_all.sh
├── src
│   └── forgetting_mlp.py
└── templates
    └── report_template.tex
```
