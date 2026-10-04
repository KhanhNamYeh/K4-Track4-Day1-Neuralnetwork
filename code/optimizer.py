"""optimizer.py — chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận được {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học. None -> không dùng. "cosine" -> cosine về `eta_min` sau total_steps bước.

    Gọi scheduler.step() SAU MỖI BƯỚC cập nhật (không phải mỗi epoch). Nếu dùng, ghi vào cột notes của bảng.
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0))
    raise ValueError(f"scheduler không hỗ trợ: {name!r}")


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    """Cắt gradient theo chuẩn L2 toàn cục và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt (tensor 0-chiều).

    - max_norm None: chỉ tính chuẩn (clip_grad_norm_ với max_norm=inf: hệ số cắt = 1, không đổi gì).
    - Trả về tensor trên cùng device (không .item()) để vòng huấn luyện không bị đồng bộ GPU ở mỗi bước.
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    limit = float("inf") if max_norm is None else float(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, limit)
