# Grokking Live (Tiny Transformer)

Sorry about the messy instructions earlier — this is the clean setup.

## Recommended: Jupyter notebook (local, no hosted site)

```bash
cd "/Users/syed/Desktop/grokking testing and erfect"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install notebook ipykernel torch matplotlib
python -m notebook
```

Open:
- `grokking_live_notebook.ipynb`

---

## Optional: script mode (terminal logs)

```bash
cd "/Users/syed/Desktop/grokking testing and erfect"
source .venv/bin/activate
python grokking_live.py
```

---

## What to expect
- Train accuracy can hit high values early.
- Test accuracy may stay low for a while.
- Later, test accuracy can jump (grokking-like behavior).

## If jump is not obvious
- Increase steps (`50k+`)
- Keep `train_frac` around `0.2–0.4`
- Keep non-zero `weight_decay`
- Try smaller model (`d_model=64`)
