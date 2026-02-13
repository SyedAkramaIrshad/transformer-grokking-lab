# Transformer Grokking Lab

A minimal, reproducible experiment to observe **grokking-like behavior** in a tiny Transformer on modular arithmetic.

## What this project shows
- Train accuracy can reach ~100% early.
- Test accuracy can stay low for many steps.
- After long training, test accuracy may jump sharply (generalization emergence).

---

## Quickstart (local Jupyter)
```bash
cd "/Users/syed/Desktop/grokking testing and erfect"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install notebook ipykernel torch matplotlib
python -m notebook
```
Open: `grokking_live_notebook.ipynb`

---

## Files
- `grokking_live_notebook.ipynb` — live plots in notebook
- `grokking_live.py` — script version
- `EXPERIMENTATION.md` — experiment design and rationale

---

## Tuning knobs
- `steps`: try `50k+` for clearer emergence
- `train_frac`: keep around `0.2–0.4`
- `weight_decay`: non-zero helps
- `d_model`: try `64` or `128`

---

## Reference
Power et al., **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**  
https://arxiv.org/abs/2201.02177
