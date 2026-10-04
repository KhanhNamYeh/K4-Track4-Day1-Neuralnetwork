"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches, N_CLASSES
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base", notes="",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine"
    seed=1,
)

TRAIN_EVAL_SUBSET = 50_000     # train loss được đo ở eval() trên một tập con CỐ ĐỊNH này
LOG_EVERY_STEPS = 50           # tần suất kiểm tra NaN/inf (tránh đồng bộ GPU ở mỗi bước)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py).
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval() (dropout tắt)."""
    model.eval()
    out = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, lấy trung bình trên MỌI phần tử (B x 7), không hệ số 1/2
               — đúng như nn.MSELoss(reduction="mean").
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        onehot = F.one_hot(y, N_CLASSES).to(torch.float32)
        return F.mse_loss(logits.float(), onehot, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận được {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1, confusion) ở chế độ eval() (dropout tắt) và no_grad.

    Cộng dồn tổng loss (reduction="sum") rồi chia cho số phần tử (N với CE, N*7 với MSE) ở cuối,
    nên kết quả không phụ thuộc batch_size.
    """
    model.eval()
    n = len(X)
    total = torch.zeros((), device=X.device, dtype=torch.float64)
    cm = torch.zeros(N_CLASSES * N_CLASSES, device=X.device, dtype=torch.int64)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name, reduction="sum").double()
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(dim=1), minlength=N_CLASSES * N_CLASSES)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    denom = n * (N_CLASSES if loss_name == "mse" else 1)
    return dict(loss=float(total.item() / denom), acc=float(np.trace(cm) / n),
                macro_f1=macro_f1_from_confusion(cm), confusion=cm)


def _fixed_train_subset(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Tập con CỐ ĐỊNH (seed riêng = 0, giống nhau ở mọi thí nghiệm) của train để đo train loss ở eval()."""
    X, y = data["X_tr"], data["y_tr"]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(X), generator=g)[:TRAIN_EVAL_SUBSET].to(X.device)
    return X[idx], y[idx]


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG); khoá thiếu lấy giá trị mặc định
        data: kết quả của data.prepare_data

    Trả về dict:
        {"cfg", "history": {epoch, train_loss, val_loss, val_acc, val_macro_f1, grad_norm, grad_norm_max,
                            clip_frac, epoch_time_s},
         "summary": {step0_loss, best_val_loss, best_epoch, final_train_loss, final_val_loss, val_acc,
                     val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged, n_steps},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}

    Lưu ý đo đạc:
      - train_loss đo ở eval() trên tập con cố định; val_* đo ở eval() trên toàn bộ val, luôn bằng FP32.
      - grad_norm = chuẩn L2 toàn cục của gradient TRƯỚC khi clip (với FP16: sau unscale_), trung bình trên
        các bước của epoch; grad_norm_max là giá trị lớn nhất (để thấy "gai"); clip_frac = tỉ lệ bước bị cắt.
      - epoch_time_s chỉ gồm phần huấn luyện (không gồm đánh giá cuối epoch), có synchronize().
      - peak_mem_MB gồm cả dữ liệu đã nằm sẵn trên GPU (~vài trăm MB), nên chênh lệch giữa các cấu hình nhỏ.
      - Không dùng X_eval ở đây: chọn epoch/cấu hình chỉ bằng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    use_cuda = device.type == "cuda"
    X_sub, y_sub = _fixed_train_subset(data)

    set_seed(cfg["seed"])
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden, cfg["dropout"], cfg["init"]).to(device)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], count_params(model)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    sched = build_scheduler(opt, cfg["scheduler"], steps_per_epoch * cfg["epochs"])

    precision = cfg["precision"]
    use_amp = precision != "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    loss_name, clip = cfg["loss"], cfg["clip_norm"]
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    step0 = evaluate(model, X_val, y_val, loss_name)["loss"]   # TRƯỚC bước cập nhật đầu tiên

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best = dict(val_loss=math.inf, epoch=None, state=None, acc=None, f1=None)
    diverged, n_steps = False, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gn_sum = torch.zeros((), device=device)
        gn_max = torch.zeros((), device=device)
        gn_cnt = torch.zeros((), device=device)
        n_clipped = torch.zeros((), device=device)
        bad = torch.zeros((), device=device, dtype=torch.bool)
        steps = 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)                      # gradient thật, TRƯỚC khi clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), clip)  # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(opt)                           # tự bỏ qua bước nếu gradient inf/NaN
                scaler.update()
            else:
                opt.step()
            if sched is not None:
                sched.step()
            ok = torch.isfinite(gn)
            gn_sum += torch.where(ok, gn, torch.zeros_like(gn))
            gn_max = torch.maximum(gn_max, torch.where(ok, gn, torch.zeros_like(gn)))
            gn_cnt += ok.float()
            if clip is not None:
                n_clipped += (ok & (gn > clip)).float()
            bad |= ~torch.isfinite(loss.detach())
            steps += 1
            if steps % LOG_EVERY_STEPS == 0 and bool(bad):
                diverged = True
                break
        n_steps += steps
        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        diverged = diverged or bool(bad)

        tr = evaluate(model, X_sub, y_sub, loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        cnt = max(float(gn_cnt), 1.0)
        for k, v in (("epoch", epoch), ("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                     ("val_acc", va["acc"]), ("val_macro_f1", va["macro_f1"]),
                     ("grad_norm", float(gn_sum) / cnt), ("grad_norm_max", float(gn_max)),
                     ("clip_frac", float(n_clipped) / max(steps, 1)), ("epoch_time_s", epoch_time)):
            hist[k].append(v)
        if math.isfinite(va["loss"]) and va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch, acc=va["acc"], f1=va["macro_f1"],
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"[{cfg['exp_id']}] ep {epoch:2d} train {tr['loss']:.4f} val {va['loss']:.4f} "
                  f"acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} gn {hist['grad_norm'][-1]:.3f} "
                  f"({epoch_time:.1f}s)")
        if diverged or not math.isfinite(va["loss"]):
            diverged = True
            if verbose:
                print(f"[{cfg['exp_id']}] loss NaN/inf -> dừng sớm (diverged)")
            break

    if best["epoch"] is None:   # chưa có epoch hữu hạn nào: báo số của epoch cuối (NaN)
        best.update(epoch=None, acc=hist["val_acc"][-1], f1=hist["val_macro_f1"][-1])
    summary = dict(
        step0_loss=step0,
        best_val_loss=best["val_loss"] if best["epoch"] is not None else float("nan"),
        best_epoch=best["epoch"],
        final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1],
        val_acc=best["acc"], val_macro_f1=best["f1"],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=(torch.cuda.max_memory_allocated() / 2**20) if use_cuda else None,
        diverged="Y" if diverged else "N", n_steps=n_steps,
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred` (đủ mọi dòng eval)."""
    import pandas as pd
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho baseline và cấu hình cuối cùng: nạp best_state (epoch val loss thấp nhất), dự đoán TOÀN BỘ eval
    ở FP32 + eval() rồi ghi predictions. Việc chấm điểm do `scripts/evaluate.py` làm (gọi từ notebook)."""
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(tuple(cfg["hidden"]), cfg["dropout"], cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
