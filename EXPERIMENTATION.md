# Experimentation Notes: Grokking with a Tiny Transformer

## What we are experimenting with
We are running a controlled experiment to observe a phenomenon similar to **grokking**:
- A model first appears to memorize the training set
- Test/generalization stays poor for a while
- After longer training, test performance can suddenly improve

This is done on a **simple synthetic task** so the learning dynamics are visible.

---

## Task
**Modular addition**:
- Input: `(a, b)` where `a, b ∈ [0, p-1]`
- Target: `(a + b) mod p`
- `p` is prime (default: `97`)

Why this task:
- Small, clean, and mathematically structured
- Known in literature to show grokking-like behavior under certain settings

---

## Model
A **tiny Transformer encoder**:
- Token embedding for `a` and `b`
- Learned position embeddings
- 2 encoder layers (default)
- Classification head over `p` classes

This is intentionally small so training is fast on MacBook Air and dynamics are easy to inspect.

---

## Training setup
Key knobs that matter:
- **Low train fraction** (`train_frac`, default `0.3`) to encourage memorization pressure
- **Weight decay** (`1e-2`) to regularize over long training
- **Long training steps** (`20k`, often `50k+` helps)
- Optimizer: AdamW

---

## What we measure live
In the notebook we track:
1. Train loss
2. Test loss
3. Train accuracy
4. Test accuracy
5. A live view of output head weights

Expected pattern (if grokking appears):
- Train accuracy rises early
- Test accuracy lags
- Later, test accuracy jumps significantly

---

## Why this is useful
This experiment helps you build intuition for:
- Memorization vs generalization
- Role of regularization and long training
- Why validation behavior can change late in training
- How model internals evolve during learning

---

## Files in this folder
- `grokking_live.py` → script version
- `grokking_live_notebook.ipynb` → live notebook version
- `EXPERIMENTATION.md` → this explanation

---

## Practical tips
If you do not see the jump clearly:
- Increase `steps` to `50k–100k`
- Keep `train_frac` around `0.2–0.4`
- Keep non-zero `weight_decay`
- Try smaller model (`d_model=64`) and rerun
