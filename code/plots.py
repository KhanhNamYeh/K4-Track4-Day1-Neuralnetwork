"""plots.py — ảnh biểu đồ là sản phẩm nộp (README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Notebook chạy trong code/ nên lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np

OPT_LABEL = {"sgd": "SGD", "sgd_momentum": "SGD+momentum", "adam": "Adam", "adamw": "AdamW"}


def _cfg_title(cfg: dict) -> str:
    clip = "none" if cfg.get("clip_norm") is None else cfg["clip_norm"]
    hidden = "-".join(str(h) for h in cfg["hidden"])
    return (f"{cfg['exp_id']}  |  {cfg['loss'].upper()}, {OPT_LABEL[cfg['optimizer']]}, lr={cfg['lr']:g}, "
            f"wd={cfg['weight_decay']:g}\nbatch={cfg['batch']}, epochs={cfg['epochs']}, hidden={hidden}, "
            f"dropout={cfg['dropout']:g}, clip={clip}, {cfg['precision']}, init={cfg['init']}, seed={cfg['seed']}")


def _clean(v):
    """NaN/inf -> NaN để matplotlib bỏ qua điểm đó thay vì làm hỏng thang trục."""
    return [x if x is not None and math.isfinite(x) else float("nan") for x in v]


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG 3 ô: (1) train/val loss, (2) val acc + val macro-F1,
    (3) grad_norm (đo TRƯỚC khi clip; đường liền = trung bình epoch, chấm = lớn nhất trong epoch).
    Đường đứt đứng đánh dấu best_epoch."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, _clean(h["train_loss"]), "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, _clean(h["val_loss"]), "s-", ms=3, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({cfg['loss'].upper()})"); ax.set_title("Train / val loss")
    finite = [x for x in _clean(h["train_loss"]) + _clean(h["val_loss"]) if not math.isnan(x)]
    if finite and max(finite) / max(min(finite), 1e-12) > 20:
        ax.set_yscale("log")

    ax = axes[1]
    ax.plot(ep, _clean(h["val_acc"]), "o-", ms=3, label="val accuracy")
    ax.plot(ep, _clean(h["val_macro_f1"]), "s-", ms=3, label="val macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("metric"); ax.set_title("Val accuracy / macro-F1")

    ax = axes[2]
    ax.plot(ep, _clean(h["grad_norm"]), "o-", ms=3, label="grad_norm (TB epoch)")
    ax.plot(ep, _clean(h["grad_norm_max"]), "^", ms=4, alpha=.6, label="grad_norm (max epoch)")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls=":", label=f"clip c={cfg['clip_norm']:g}")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂ trước clip"); ax.set_title("Chuẩn gradient")
    vals = [x for x in _clean(h["grad_norm"]) + _clean(h["grad_norm_max"]) if not math.isnan(x) and x > 0]
    if vals:
        ax.set_yscale("log")

    for ax in axes:
        if s.get("best_epoch"):
            ax.axvline(s["best_epoch"], color="gray", ls="--", lw=1, label=f"best epoch = {s['best_epoch']}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    if s.get("diverged") == "Y":
        fig.text(0.5, 0.01, "DIVERGED (loss NaN/inf) — huấn luyện dừng sớm", ha="center", color="red")
    fig.suptitle(_cfg_title(cfg), fontsize=9, y=1.04)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "", labels: list[str] | None = None) -> None:
    """Vẽ chồng chỉ số `metric` (hoặc danh sách các chỉ số, mỗi chỉ số một ô) của nhiều thí nghiệm,
    mỗi thí nghiệm một đường, chú thích bằng exp_id (hoặc `labels`)."""
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for i, r in enumerate(results):
            h = r["history"]
            lab = labels[i] if labels else r["cfg"]["exp_id"]
            ax.plot(h["epoch"], _clean(h[m]), "o-", ms=2.5, lw=1.4, label=lab)
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(m); ax.grid(alpha=.3)
        vals = [x for r in results for x in _clean(r["history"][m]) if not math.isnan(x)]
        if m in ("grad_norm", "grad_norm_max") and vals and min(vals) > 0:
            ax.set_yscale("log")
        if m.endswith("loss") and vals and max(vals) / max(min(vals), 1e-12) > 20:
            ax.set_yscale("log")
        if m in ("val_macro_f1", "val_acc") and vals:
            lo = np.percentile(vals, 5)   # đoạn đầu rất thấp làm dẹt phần đáng xem
            ax.set_ylim(max(0.0, min(lo, 0.9 * max(vals)) - 0.02), min(1.0, max(vals) + 0.01))
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
