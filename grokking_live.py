import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    prime: int = 97                # modulus (classic grokking uses a prime)
    train_frac: float = 0.3        # low-data regime helps grokking appear
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 2
    dropout: float = 0.0
    batch_size: int = 256
    steps: int = 20000
    eval_every: int = 100
    log_every: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


cfg = CFG()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data: (a + b) mod p
# -----------------------------

def build_dataset(p: int):
    xs = []
    ys = []
    for a in range(p):
        for b in range(p):
            xs.append((a, b))
            ys.append((a + b) % p)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# -----------------------------
# Tiny transformer
# -----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, nlayers: int, dropout: float, out_classes: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)  # sequence length = 2 (a,b)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_classes)

    def forward(self, x):
        # x: [B,2]
        h = self.token_emb(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])  # use token-2 representation
        return self.head(h)


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.size(0)
    return loss_sum / total, correct / total


def batch_sample(x, y, bs):
    idx = torch.randint(0, len(x), (bs,), device=x.device)
    return x[idx], y[idx]


def weight_l2(model):
    s = 0.0
    for p in model.parameters():
        s += (p.detach() ** 2).sum().item()
    return math.sqrt(s)


def main():
    set_seed(cfg.seed)
    print(f"Using device: {cfg.device}")

    x, y = build_dataset(cfg.prime)
    n = len(x)
    perm = torch.randperm(n)
    train_n = int(cfg.train_frac * n)
    tr_idx, te_idx = perm[:train_n], perm[train_n:]

    x_train, y_train = x[tr_idx], y[tr_idx]
    x_test, y_test = x[te_idx], y[te_idx]

    x_train, y_train = x_train.to(cfg.device), y_train.to(cfg.device)
    x_test, y_test = x_test.to(cfg.device), y_test.to(cfg.device)

    model = TinyTransformer(
        vocab_size=cfg.prime,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        nlayers=cfg.nlayers,
        dropout=cfg.dropout,
        out_classes=cfg.prime,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    writer = SummaryWriter(log_dir="runs/grokking_live")

    print("\nLive logs:")
    print("step | train_loss train_acc | test_loss test_acc | weight_l2")
    print("-" * 70)

    for step in range(1, cfg.steps + 1):
        model.train()
        xb, yb = batch_sample(x_train, y_train, cfg.batch_size)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == 1:
            tr_loss, tr_acc = evaluate(model, x_train, y_train)
            te_loss, te_acc = evaluate(model, x_test, y_test)
            wl2 = weight_l2(model)

            print(f"{step:5d} | {tr_loss:9.4f} {tr_acc:8.4f} | {te_loss:8.4f} {te_acc:8.4f} | {wl2:9.2f}")

            writer.add_scalar("loss/train", tr_loss, step)
            writer.add_scalar("loss/test", te_loss, step)
            writer.add_scalar("acc/train", tr_acc, step)
            writer.add_scalar("acc/test", te_acc, step)
            writer.add_scalar("weights/l2", wl2, step)

            # Log one representative weight matrix image periodically
            if step % (cfg.eval_every * 5) == 0:
                W = model.head.weight.detach().float().cpu()  # [p, d_model]
                Wn = (W - W.min()) / (W.max() - W.min() + 1e-8)
                writer.add_image("weights/head_weight", Wn.unsqueeze(0), step)

    writer.close()
    print("\nDone. Open TensorBoard to watch curves + weight image updates:")
    print("tensorboard --logdir runs/grokking_live")


if __name__ == "__main__":
    main()
