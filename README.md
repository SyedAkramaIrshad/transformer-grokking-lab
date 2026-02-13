# Grokking Live (Tiny Transformer)

This script trains a tiny transformer on modular addition and logs live metrics + weight snapshots.

## 1) Install deps
```bash
pip install torch tensorboard
```

## 2) Run training
```bash
cd "/Users/syed/Desktop/grokking testing and erfect"
python grokking_live.py
```

## 3) View live charts (loss/acc/weight-l2 + weight image)
In a second terminal:
```bash
cd "/Users/syed/Desktop/grokking testing and erfect"
tensorboard --logdir runs/grokking_live
```
Then open: http://localhost:6006

## What to watch
- Train accuracy can go high earlier.
- Test accuracy may stay low for long time, then jump (grokking-like behavior).
- Weight L2 and weight image evolution help you "see" parameters changing.

## If grokking does not appear immediately
Try:
- Increase `steps` (e.g., 50k+)
- Keep `train_frac` low (0.2-0.4)
- Keep non-zero `weight_decay`
- Reduce model size slightly
