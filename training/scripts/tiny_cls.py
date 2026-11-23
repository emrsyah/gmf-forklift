"""CNN kecil biner (dari nol, tanpa bobot pretrained) untuk potongan grayscale: APD (A8) dan mata (A9).

CPU laptop cukup. Export ONNX (input "x" [1,1,S,S] float 0..1, output "logits" [1,2]; kelas 1 = pelanggaran/tertutup).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_net(blocks: int = 4, width: int = 16):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))
    chans = [1] + [width * 2 ** i for i in range(blocks)]
    layers = [blk(chans[i], chans[i + 1]) for i in range(blocks)]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(chans[-1], 2))


def augment(xb, blur_max_scale: float = 0.0):
    """Kecerahan/kontras/noise (kamera kabin redup) + opsional perkecil-perbesar (wajah kecil)."""
    import torch
    import torch.nn.functional as F

    n, _, h, w = xb.shape
    if blur_max_scale > 0:
        s = float(torch.empty(1).uniform_(blur_max_scale, 1.0))
        if s < 0.95:
            small = F.interpolate(xb, size=(max(4, int(h * s)), max(4, int(w * s))), mode="area")
            xb = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    a = torch.empty(n, 1, 1, 1).uniform_(0.6, 1.4)
    b = torch.empty(n, 1, 1, 1).uniform_(-0.15, 0.15)
    g = torch.empty(n, 1, 1, 1).uniform_(0.7, 1.4)
    xb = xb.clamp(1e-4, 1) ** g
    xb = (xb - 0.5) * a + 0.5 + b
    xb = xb + torch.randn_like(xb) * torch.empty(n, 1, 1, 1).uniform_(0, 0.05)
    return xb.clamp(0, 1)


def balanced_acc(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    r = [np.mean(pred[y == c] == c) for c in (0, 1) if (y == c).any()]
    return float(np.mean(r)) if r else 0.0


def stats(y, pred) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"recall_pos": tp / max(1, tp + fn), "false_pos_rate": fp / max(1, fp + tn), "bal_acc": balanced_acc(y, pred),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def predict(net, X: np.ndarray, batch: int = 512) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch, None].astype(np.float32) / 255)
            out.append(torch.softmax(net(xb), 1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(name: str, Xtr, ytr, Xva, yva, epochs: int, out: Path, seed: int = 0, blocks: int = 4, width: int = 16,
          blur_max_scale: float = 0.0, batch: int = 128, flip: bool = True):
    """Latih, simpan state terbaik (bal-acc val), export <out>/<name>.onnx. -> (net, best_val_bal_acc)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    net = make_net(blocks, width)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * ((len(Xtr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(1)), dtype=torch.float32)   # imbang kelas
    xt = torch.from_numpy(Xtr[:, None].astype(np.float32) / 255)
    yt = torch.from_numpy(np.asarray(ytr)).long()
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            xb = augment(xt[idx], blur_max_scale)
            if flip and torch.rand(1).item() < 0.5:
                xb = xb.flip(3)
            loss = F.cross_entropy(net(xb), yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(net, Xva)
        bacc = balanced_acc(yva, pv >= 0.5)
        if bacc > best:
            best, best_state = bacc, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  [{name}] epoch {ep + 1}/{epochs} loss {loss.item():.3f} val bal-acc {bacc:.3f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 1, Xtr.shape[1], Xtr.shape[2]), str(out / f"{name}.onnx"),
                      input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}})
    return net, best


def pick_threshold(yva, pv, max_fpr: float) -> float:
    """Ambang di VAL: bal-acc terbaik dengan false positive rate <= max_fpr (alarm palsu)."""
    cands = [(balanced_acc(yva, pv >= t), float(t)) for t in np.arange(0.3, 0.96, 0.05)
             if stats(yva, pv >= t)["false_pos_rate"] <= max_fpr]
    return round(max(cands)[1], 2) if cands else 0.5
