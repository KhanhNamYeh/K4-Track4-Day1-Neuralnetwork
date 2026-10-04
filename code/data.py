"""data.py — nạp dữ liệu đã chia sẵn, tách validation, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_tr, y_tr = tr["X"], tr["y"]
    X_ev, y_ev, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_tr, y_tr), (X_ev, y_ev)):
        assert X.ndim == 2 and X.shape[1] == N_FEATURES and X.dtype == np.float32, (X.shape, X.dtype)
        assert y.shape == (X.shape[0],) and y.dtype == np.int64, (y.shape, y.dtype)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1
    assert len(row_id) == len(y_ev)
    return X_tr, y_tr, X_ev, y_ev, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val. Cùng seed và val_fraction cho mọi thí nghiệm.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Mean và std của N_NUMERIC cột đầu, tính CHỈ trên phần train (sau khi tách val).

    Vì sao không tính trên val/eval/toàn bộ dữ liệu: thống kê đó chứa thông tin của mẫu mà mô hình
    không được phép thấy khi huấn luyện (rò rỉ); val/eval phải được biến đổi bằng đúng tham số của train
    thì mới mô phỏng đúng tình huống dữ liệu mới chưa từng thấy.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # tránh chia cho 0 nếu một cột hằng
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu (x - mean) / std; 44 cột nhị phân giữ nguyên. Không sửa X tại chỗ."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict: X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensor trên device, y int64),
    eval_row_id (numpy), mean, std (thống kê chuẩn hoá của train), majority_acc_val.
    """
    X_full, y_full, X_ev, y_ev, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)             # CHỈ trên X_tr
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)   # cùng mean/std của train
    X_ev = apply_standardizer(X_ev, mean, std)

    def to_t(X, y):
        return (torch.tensor(X, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.int64, device=device))

    X_tr_t, y_tr_t = to_t(X_tr, y_tr)
    X_val_t, y_val_t = to_t(X_val, y_val)
    X_ev_t, y_ev_t = to_t(X_ev, y_ev)

    majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())
    majority_acc = float((y_val == majority).mean())
    print(f"train {len(y_tr):,} | val {len(y_val):,} | eval {len(y_ev):,}")
    print(f"'luôn đoán lớp đa số' (lớp {majority}) trên val: acc = {majority_acc:.4f}")
    return dict(X_tr=X_tr_t, y_tr=y_tr_t, X_val=X_val_t, y_val=y_val_t,
                X_eval=X_ev_t, y_eval=y_ev_t, eval_row_id=row_id,
                mean=mean, std=std, majority_acc_val=majority_acc)


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Xáo bằng torch.randperm mỗi epoch. Batch cuối có thể nhỏ hơn batch_size; ta GIỮ nó (không bỏ mẫu nào).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
