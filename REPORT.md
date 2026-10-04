# Báo cáo Lab Day 1 — [điền họ tên] — 523H0148

## 1. Thiết lập

- Môi trường: huấn luyện trên Google Colab, GPU Tesla T4, PyTorch 2.11. Mọi lần chạy được lưu ở `results/<exp_id>.json`; output trong `code/lab.ipynb` được tạo khi chạy lại notebook, lúc đó `run_and_log` đọc lại các kết quả này (dòng ghi "[đọc lại từ results/]") và các ô kiểm tra Part 0–1 chạy trực tiếp (máy RTX 4050, PyTorch 2.14).
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. Chuẩn hoá 10 cột số chỉ bằng thống kê của phần train.
- Model: `M-base` (54→256→128→7, 47 879 tham số, có `assert`). Baseline: CE, SGD+momentum 0,9, **lr = 0,3** (chọn bằng val trong lưới {0,01; 0,03; 0,1; 0,3; 1}), batch 512, 20 epoch, He init, không dropout/clip, FP32.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0,4876 (macro-F1 ≈ 0,094).
- Các chủ đề đã thử: ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init (57 lần chạy, một dòng mỗi lần trong `experiments.xlsx`).

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (8, 7) với lô (8, 54) |
| Loss bước 0 (so với ln 7 = 1,946) | He: 2,21 / 2,27 / 1,98 / 1,90 / 2,31 (seed 0–4); `zeros`: 1,9459 |
| Quá khớp 20 mẫu: loss cuối | 1,97 → 7,4·10⁻⁴ sau 300 bước (Adam 1e-3), accuracy 20/20 |
| Mọi tham số có gradient khác 0 | ☑ có: ‖∇W1‖ 0,45; ‖∇b1‖ 0,27; ‖∇W2‖ 1,76; ‖∇b2‖ 0,38; ‖∇W3‖ 1,94; ‖∇b3‖ 0,55 |
| Baseline, số seed đã chạy | 3 (`base-s1..3`) |
| Baseline: val acc (TB ± σ) | 0,9118 ± 0,0027 |
| Baseline: val macro-F1 (TB ± σ) | 0,8607 ± 0,0030 (0,8573; 0,8618; 0,8630) |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0,0060** (val macro-F1). σ chỉ ước lượng từ 3 seed nên là con số thô.

Loss bước 0: He init giữ độ lớn kích hoạt cả ở lớp logit (std logit ≈ 0,58), nên logit ban đầu lệch khỏi 0. CE lồi theo logit nên kỳ vọng loss lớn hơn ln 7. Mức 1,9–2,3 vẫn cùng bậc ln 7, không phải dấu hiệu lỗi nhãn hay softmax hai lần. Loss đúng bằng 1,946 chỉ khi logit = 0 (`zeros`; `normal` cho std logit 0,0003 nên loss cũng ≈ 1,946).

Baseline (`figures/base-s1.png`): train/val loss giảm đều đến epoch 19–20 (best epoch 19), val acc 0,909 ≫ 0,4876. Khoảng cách val−train nhỏ (0,234 so với 0,206) và đường val còn đi xuống ⇒ **chưa quá khớp, chưa hội tụ hẳn**. `grad_norm` trung bình mỗi epoch ≈ 0,34–0,38, có một gai 2,8 ở epoch 1.

## 3. Kết quả theo chủ đề

So sánh trên val. `Δ` = val macro-F1 − 0,8607; "vượt nhiễu" khi |Δ| > 0,0060. Mỗi cấu hình chạy 1 seed (trừ baseline và cấu hình cuối).

### 3.1 Hàm mất mát — CE vs MSE
- Dự đoán: MSE trên logit cho gradient nhỏ hơn CE (lấy trung bình trên B×7 phần tử, không phạt mạnh khi tự tin sai) ⇒ học chậm hơn và macro-F1 thấp hơn.
- Kết quả: `loss-mse-lr0.3` F1 **0,7726**, acc 0,8803; `loss-mse-lr0.1` 0,7277; `loss-mse-lr1` 0,5991. CE (`base-s1`) đạt 0,8573, trung bình 3 seed 0,8607. Δ = −0,088 ≫ 2σ. Tốc độ: macro-F1 ở epoch 1/5/9/13/17 của MSE là 0,48/0,66/0,70/0,74/0,77, của CE là 0,60/0,80/0,82/0,84/0,85. Ảnh: `figures/compare_loss.png`. Không so giá trị loss vì CE và MSE khác thang đo.
- Giải thích: gradient của CE theo logit là `p−y`, lớn khi sai nặng và không bão hoà. Gradient của MSE là `2(z−y)/(7B)`, nhỏ hơn và chia đều cho mọi logit nên lớp hiếm ít được kéo lên. Tăng lr lên 1,0 không bù được, còn làm huấn luyện bất ổn (grad_norm max 14). Khớp dự đoán.

### 3.2 Bộ tối ưu hoá
- Dự đoán: SGD thuần cần lr khoảng 10× SGD+momentum; Adam/AdamW ít nhạy lr hơn; khi chỉnh lr công bằng thì khoảng cách thu hẹp; AdamW(wd=0) ≡ Adam.
- Mỗi bộ ở lr tốt nhất của nó:

| exp_id | lr | val macro-F1 | best ep | Δ vs 0,8607 | vượt nhiễu? |
|---|---|---|---|---|---|
| `opt-sgd-lr1` | 1,0 | 0,8344 | 18 | −0,026 | Có (tệ hơn) |
| `base-s1` (SGD+momentum) | 0,3 | 0,8573 | 19 | −0,003 | Không |
| `opt-adam-lr3e-3` | 3e-3 | 0,8677 | 19 | +0,007 | Có, sát ngưỡng |
| `opt-adamw-lr3e-3` (wd 0,01) | 3e-3 | 0,8646 | 18 | +0,004 | Không |

- Độ nhạy với lr (`figures/compare_optimizer_lr_sensitivity.png`, `figures/compare_optimizer.png`):
  - SGD, lr {0,3; 1; 3}: 0,816 / 0,834 / 0,616.
  - SGD+momentum, lr {0,01; 0,03; 0,1; 0,3; 1}: 0,761 / 0,817 / 0,839 / 0,857 / 0,773.
  - Adam, lr {3e-4; 1e-3; 3e-3; 1e-2}: 0,788 / 0,846 / 0,868 / 0,864.
  - AdamW, cùng lưới: 0,786 / 0,848 / 0,865 / 0,849.
- SGD thuần tốt nhất ở lr ≈ 3× SGD+momentum (1,0 so với 0,3), không phải 10× như dự đoán. Adam giữ ≥ 0,845 trên 3 giá trị lr liền nhau; SGD+momentum chỉ đạt mức đó ở 1 giá trị.
- `opt-adamw-wd0-lr3e-3` và `opt-adam-lr3e-3` cho cùng số đến 4 chữ số (val loss 0,2145; macro-F1 0,8677), đúng lý thuyết.
- Giải thích: Adam chia bước cho √v̂ nên mỗi tham số có bước hiệu dụng riêng. SGD+momentum dùng một lr chung nên nhạy với độ lớn gradient.

### 3.3 Hyper-parameter
Số bước cập nhật mỗi epoch: batch 128 → 2 905; batch 512 → 727; batch 2048 → 182.

| exp_id | thay đổi | val macro-F1 | best ep | Δ | vượt nhiễu |
|---|---|---|---|---|---|
| `hp-b128-lr0.3` | batch 128, giữ lr 0,3 | 0,7951 | 14 | −0,066 | Có (tệ) |
| `hp-b128-lr0.075` | batch 128, lr 0,3/4 | 0,8647 | 20 | +0,004 | Không |
| `hp-b2048-lr0.3` | batch 2048 | 0,8338 | 18 | −0,027 | Có (tệ) |
| `hp-b2048-lr1.2` | batch 2048, lr ×4 | 0,0936 | 11 | — | sụp đổ |
| `hp-wide` | M-wide | 0,8747 | 20 | +0,014 | Có (tốt) |
| `hp-deep` | M-deep | 0,8413 | 20 | −0,019 | Có (tệ) |
| `hp-wd1e-4` | weight_decay 1e-4 | 0,8123 | 18 | −0,048 | Có (tệ) |
| `hp-ep40` | 40 epoch | 0,8733 | 32 | +0,013 | Có (tốt) |

Ảnh: `figures/compare_hparam_batch.png`, `figures/compare_hparam_other.png`.
- Batch 128 với lr 0,3 dao động (best epoch 14). Hạ lr theo tỉ lệ lô (0,075) thì hoạt động. Theo chiều tăng, batch 2048 với lr 1,2 và không khởi động làm mạng sụp về đoán lớp đa số.
- `M-wide` và 40 epoch tốt hơn, đúng dự đoán vì baseline chưa hội tụ.
- **Không như dự đoán:** `M-deep` tệ hơn baseline, còn `wd=1e-4` gây hại nhiều dù mạng chưa quá khớp. Lý do khả dĩ: lr 0,3 chưa hợp với mạng 3 lớp ẩn, và L2 làm co trọng số khi mạng còn chưa khớp. Đây là phỏng đoán, chưa kiểm chứng.
- Thời gian/epoch theo tỉ lệ: batch 128 ≈ 4× batch 512; batch 2048 ≈ 0,26×.

### 3.4 Dropout
- Dự đoán: baseline chưa quá khớp nên dropout không giúp; q càng lớn càng tệ.
- Kết quả (`figures/compare_dropout.png`, train loss đo ở `eval()`): khoảng cách val−train loss là 0,028 ở baseline, còn 0,015 / 0,006 / 0,004 với q = 0,1 / 0,3 / 0,5. Nhưng val macro-F1 tụt còn 0,8385 / 0,7824 / 0,5784; cả ba đều vượt nhiễu theo hướng tệ hơn.
- Giải thích: dropout thu hẹp khoảng cách bằng cách làm train loss tệ đi (underfitting). Mô hình không quá khớp nên không có gì để chữa.

### 3.5 Gradient clipping
Nhìn `grad_norm` của baseline trước khi chọn c: trung bình mỗi epoch ≈ 0,34–0,38; sau epoch 1 giá trị lớn nhất ≤ 0,86; epoch 1 có gai 2,8. Do đó chọn c ∈ {1,0; 0,5; 0,3}.

| exp_id | lr | c | tỉ lệ bước bị cắt (TB) | val macro-F1 | so sánh |
|---|---|---|---|---|---|
| `clip-c1-lr0.3` | 0,3 | 1,0 | 0,1% | 0,8534 | trong nhiễu |
| `clip-c0.5-lr0.3` | 0,3 | 0,5 | 1,4% | 0,8481 | Δ −0,013 |
| `clip-c0.3-lr0.3` | 0,3 | 0,3 | 99,5% | 0,8311 | Δ −0,030 |
| `opt-sgdm-lr1` (không clip) | 1,0 | — | 0 | 0,7732 | đối chứng |
| `clip-c1-lr1` | 1,0 | 1,0 | ≈ 0% | 0,8106 | +0,037 so với không clip |
| `clip-c0.3-lr1` | 1,0 | 0,3 | 9,6% | 0,8297 | +0,057 so với không clip |
| `clip-none-lr3` | 3,0 | — | 0 | 0,0936 (sụp; gn max 7 813) | đối chứng |
| `clip-c1-lr3` | 3,0 | 1,0 | 0,1% | 0,0936 (sụp) | không cứu được |
| `clip-c0.3-lr3` | 3,0 | 0,3 | 6,3% | 0,1875 (best ep 2) | không cứu được |

Ảnh: `figures/compare_clipping_lr0.3.png`, `figures/compare_clipping_highlr.png`.
- Ở lr ổn định (0,3): c lớn hơn vài lần grad_norm điển hình gần như không kích hoạt, chênh lệch nằm trong nhiễu. c nhỏ hơn mức điển hình kích hoạt ở mọi bước, tương đương giảm lr hiệu dụng, nên kém hơn. Đúng dự đoán.
- Ở lr 1,0: clipping cứu được một phần (0,773 → 0,81 và 0,83, vượt 2σ), nhưng vẫn kém baseline lr 0,3.
- **Không như dự đoán:** ở lr 3,0 clipping không cứu được. Giải thích khả dĩ: clip chặn chuẩn gradient chứ không chặn bước cập nhật lr·c/(1−μ), mà bước này vẫn quá lớn. Sau gai đầu tiên, val loss đứng ở ≈ 1,209, xấp xỉ entropy của phân phối lớp (≈ 1,20), tức mạng chỉ còn học tần suất lớp.

### 3.6 Mixed precision
- Dự đoán: với mạng nhỏ, thời gian bị chi phối bởi chi phí gọi kernel nên FP16 không nhanh hơn; BF16 trên T4 (không có BF16 phần cứng) còn chậm hơn; bộ nhớ gần như không đổi; độ chính xác ≈ FP32; chỉ mạng rất rộng (2048-2048) mới hưởng lợi từ FP16.
- Độ chính xác (val macro-F1): `amp-fp16` 0,8502 và `amp-bf16` 0,8506, so với `base-s1` 0,8573. Chênh khoảng −0,007, cỡ ngưỡng 2σ với 1 seed ⇒ **không kết luận được** là AMP làm giảm chất lượng. Mạng 2048-2048 (3 epoch): FP32 0,8082; FP16 0,7993; BF16 0,8013.

| exp_id | s/epoch | peak mem (MB) |
|---|---|---|
| `base-s1` (FP32) | 2,12 | 174 |
| `amp-fp16` | 3,04 | 181 |
| `amp-bf16` | 2,59 | 182 |
| `amp-big-fp32` | 6,93 | 358 |
| `amp-big-fp16` | 3,42 | 375 |
| `amp-big-bf16` | 9,71 | 391 |

- Thời gian/epoch được đo khi GPU không dành riêng cho tiến trình huấn luyện, nên chỉ dùng để so sánh tương đối. Hướng của kết quả rõ ràng: với `M-base`, FP16/BF16 không nhanh hơn FP32; với mạng 2048-2048, FP16 nhanh khoảng 2×; BF16 chậm nhất trên T4.
- Giải thích: FP16 có khoảng giá trị hẹp (dễ tràn hoặc underflow) nên cần `GradScaler` nhân loss với hệ số `s`. BF16 có 8 bit mũ như FP32 (khoảng giá trị rộng) nhưng ít bit mantissa, nên không cần scaler. Bộ nhớ không giảm vì phần lớn là dữ liệu nằm sẵn trên GPU, và autocast còn tạo thêm bản sao đã cast.

### 3.7 Khởi tạo tham số

Bước 0, seed 1, lô 4 096 mẫu val; std đo ở đầu ra mỗi `nn.Linear`, trước ReLU:

| init | loss bước 0 | std L1 | std L2 | std logit | ‖∇W1‖ | ‖∇W2‖ | ‖∇W3‖ | ‖∇b3‖ |
|---|---|---|---|---|---|---|---|---|
| zeros | 1,9459 | 0 | 0 | 0 | 0 | 0 | 0 | 0,497 |
| normal (0,01) | 1,9460 | 0,034 | 0,0038 | 0,0003 | 0,004 | 0,007 | 0,010 | 0,497 |
| xavier_normal | 2,0222 | 0,276 | 0,218 | 0,192 | 0,361 | 0,729 | 0,599 | 0,518 |
| he | 2,2691 | 0,661 | 0,640 | 0,577 | 0,507 | 2,021 | 1,960 | 0,572 |
| default | 1,9830 | 0,274 | 0,113 | 0,059 | 0,078 | 0,347 | 0,339 | 0,508 |

| exp_id | val macro-F1 | val acc | Δ |
|---|---|---|---|
| `init-zeros` | 0,0936 | 0,4876 | sụp |
| `init-normal` | 0,8609 | 0,9137 | +0,000 |
| `init-xavier` | 0,8502 | 0,9096 | −0,011 (sát ngưỡng) |
| `init-default` | 0,8637 | 0,9145 | +0,003 |
| `base-s1` (he) | 0,8573 | 0,9090 | −0,003 |

Ảnh: `figures/compare_init.png`, `figures/compare_init_activation_depth30.png`.
- `zeros` đúng dự đoán: gradient của W1, W2, W3 bằng 0 tuyệt đối, chỉ b3 có gradient (do ReLU(0) = 0 và đối xứng giữa các nơ-ron). Mạng chỉ học được bias lớp cuối: val loss ≈ 1,205, xấp xỉ entropy phân phối lớp; acc 0,4876; macro-F1 0,094.
- `normal`: std co mạnh qua mỗi lớp (0,034 → 0,004 → 0,0003), gradient của W chỉ ≤ 0,01 ở bước 0, nhưng mạng 3 lớp vẫn học tới 0,861. Xavier co nhẹ (thiếu hệ số 2 bù cho ReLU), còn He giữ std ổn định.
- Ở mạng 30 lớp ẩn, khác biệt rất rõ: std của `normal` về ~0 trước lớp 25, `xavier` giảm dần tới ~10⁻⁵, `he` giữ ≈ 1. Ở mạng 3 lớp của lab, khác biệt giữa các cách khởi tạo hợp lý nằm trong khoảng ±0,011.

## 4. Đánh giá cuối trên tập eval

**Cấu hình cuối cùng** được chọn theo val, thêm từng yếu tố một (seed 1):

| exp_id | cấu hình | val macro-F1 |
|---|---|---|
| `fin-c1-adam-base-cos` | Adam + cosine, M-base | 0,8826 |
| `fin-c2-adam-wide` | Adam, M-wide | 0,8914 |
| `fin-c3-adam-wide-cos` | Adam + cosine, M-wide | 0,9139 |
| **`fin-c4-adam-wide-cos-ep40`** | **Adam lr 3e-3 + cosine về 0, M-wide (512-256), 40 epoch** | **0,9267** |

Các yếu tố còn lại giữ như baseline (He, CE, batch 512, không dropout/clip, FP32). So với baseline, cấu hình này đổi nhiều yếu tố cùng lúc và dùng gấp đôi số epoch.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (`base-s1`) | 1 | 0,8573 | **0,8565** | 0,9089 |
| Cấu hình cuối cùng (`fin-c4-adam-wide-cos-ep40`) | 1 | 0,9267 | **0,9294** | 0,9531 |

Nguồn số: `results/eval/baseline_eval_result.json` và `eval_result.json` (đầu ra của `scripts/evaluate.py`). `predictions_eval.csv` là dự đoán của cấu hình cuối, seed 1.
- Cải thiện trên eval: **+0,073** macro-F1, lớn hơn nhiều so với 2σ = 0,006. Trên val, 3 seed của cấu hình cuối đạt 0,9267 / 0,9254 / 0,9272 = 0,9264 ± 0,0010, so với baseline 0,8607 ± 0,0030. Eval chỉ được chấm một lần cho mỗi cấu hình, sau khi cấu hình đã chốt.
- Val và eval rất gần nhau: baseline 0,8573 so với 0,8565; cấu hình cuối 0,9267 so với 0,9294.

### 4.1 Phân tích lỗi theo lớp (cấu hình cuối, eval)

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 Spruce/Fir | 42 368 | 0,9524 | 0,9472 | 0,9498 |
| 1 Lodgepole Pine | 56 661 | 0,9573 | 0,9632 | 0,9602 |
| 2 Ponderosa Pine | 7 151 | 0,9539 | 0,9547 | 0,9543 |
| 3 Cottonwood/Willow | 549 | 0,8969 | 0,8561 | 0,8760 |
| 4 Aspen | 1 899 | 0,9086 | 0,8794 | 0,8938 |
| 5 Douglas-fir | 3 473 | 0,9200 | 0,9110 | 0,9155 |
| 6 Krummholz | 4 102 | 0,9570 | 0,9559 | 0,9565 |

- Lớp khó nhất là **lớp 3** (F1 = 0,876, chỉ 549 mẫu ≈ 0,47%). Lớp này bị nhầm nhiều nhất sang **lớp 2** (9,8% mẫu thật) và lớp 5 (4,6%). Lớp 4 (F1 0,894) bị kéo về lớp đa số 1 (9,5%). Ma trận nhầm lẫn: `figures/error_analysis_confusion.png`.
- Lý giải: lớp 3 và 4 rất ít mẫu nên đóng góp nhỏ vào loss, trong khi lớp 1 áp đảo (48,8%). Cặp 3 và 2 có thể có địa hình tương tự (nhận định, chưa kiểm chứng). Baseline yếu nhất ở lớp 4 (F1 0,760; 26% bị đoán thành lớp 1); cấu hình cuối cải thiện mạnh nhất ở chính các lớp hiếm.
- Hướng cải thiện sẽ thử: trọng số lớp trong CE hoặc oversampling lớp 3 và 4.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng khi chỉnh lr công bằng?** Adam (3e-3) 0,868 ≳ AdamW 0,865 ≈ SGD+momentum 0,861 > SGD 0,834. Chênh Adam so với SGD+momentum là 0,007, chỉ vừa sát ngưỡng 2σ (0,006) với 1 seed, nên chưa đủ để khẳng định Adam thắng. Nếu không chỉnh lr thì kết luận lệch mạnh: ví dụ so Adam lr 1e-3 (0,846) với SGD+momentum lr 0,01 (0,761), Adam "thắng" tới 0,085.
2. **Dropout có giúp khi chưa quá khớp?** Không: mọi q đều làm tệ hơn (mục 3.4). Chỉ nên dùng khi val loss bắt đầu tăng hoặc khoảng cách val−train lớn dần.
3. **Gradient clipping giải quyết vấn đề gì?** Chặn các bước có gradient đột ngột lớn. Bằng chứng: ở lr 1,0, clip nâng macro-F1 từ 0,773 lên 0,81–0,83. Clip không chữa được lr quá cao nói chung (lr 3,0 vẫn sụp), và hơi gây hại khi kích hoạt ở mọi bước (c = 0,3 ở lr 0,3).
4. **Mixed precision có nhanh hơn không?** Không, với `M-base`: mạng quá nhỏ nên chi phí gọi kernel và cast lấn át lợi ích của Tensor Core. Với mạng 2048-2048 thì FP16 nhanh khoảng 2×.
5. **Vì sao khởi tạo toàn 0 hỏng? He khác Xavier ở đâu?** Toàn 0 thì kích hoạt bằng 0, ReLU'(0) = 0 và mọi nơ-ron đối xứng, nên gradient của mọi W bằng 0 và chỉ b3 học được. He dùng Var = 2/n_in (bù việc ReLU triệt nửa phương sai), Xavier dùng 2/(n_in+n_out). Khác biệt này quan trọng ở mạng sâu (30 lớp: std của Xavier giảm tới ~10⁻⁵ còn He giữ ≈ 1); ở mạng 3 lớp thì không đo được khác biệt.
6. **Loss không giảm sau 2 000 bước: 3 phép kiểm tra đầu tiên.**
   - (i) Loss bước 0 ≈ ln C, để bắt lỗi nhãn, softmax hai lần hoặc khởi tạo sai. Ở đây đo được 1,9–2,3 với He và đúng 1,946 khi logit = 0.
   - (ii) Quá khớp được một lô nhỏ khi tắt mọi chính quy hoá, để bắt lỗi vòng lặp huấn luyện hoặc optimizer. Ở đây 20 mẫu đạt loss 7·10⁻⁴.
   - (iii) In chuẩn gradient từng tham số và `grad_norm`. Gradient bằng 0 như ở `zeros` cho thấy mạng không thể học; gai hàng nghìn như ở lr 3,0 cho thấy lr quá cao.
   
   Kèm theo đó, luôn ghi train và val loss đo ở `eval()`.

## 6. Hạn chế và điều bất ngờ

- **Kết quả khác dự đoán:** `M-deep` tệ hơn baseline; `weight_decay=1e-4` gây hại nhiều; clipping không cứu được lr 3,0; SGD thuần cần lr ≈ 3× (không phải 10×) so với SGD+momentum; loss bước 0 của He là 1,9–2,3 chứ không ≈ ln 7.
- **Điểm có thể làm kết luận sai:**
  - Hầu hết cấu hình chỉ chạy 1 seed, và σ ước lượng từ 3 seed baseline; nhiều chênh lệch nằm sát ngưỡng 2σ (Adam, xavier, AMP).
  - Cùng số epoch nhưng số bước khác nhau ở nhóm batch; cấu hình cuối đổi nhiều yếu tố cùng lúc và dùng gấp đôi số epoch.
  - Lưới lr thưa (bước ×3).
  - Các cơ chế nêu cho `M-deep`, lr 3,0 và cặp lớp 3↔2 là phỏng đoán.
  - Thời gian/epoch đo khi GPU không dành riêng cho tiến trình huấn luyện, nên chỉ dùng để so sánh tương đối.
- **Nếu có thêm thời gian:** thêm seed cho các so sánh sát ngưỡng; đo lại thời gian trên GPU dành riêng; kiểm tra tỉ lệ ReLU chết ở lr 3,0 và ở `M-deep`; thử trọng số lớp cho lớp 3 và 4.

## 7. Phụ lục

- File nộp: `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/` (57 ảnh `<exp_id>.png`, các ảnh `compare_*.png` và `error_analysis_confusion.png`), `results/` (57 file `<exp_id>.json` và `eval/baseline_eval_result.json`), `code/` (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
- Thời gian chạy: mỗi thí nghiệm 0,5–3 phút trên T4 (batch 128 và mạng lớn lâu hơn); cả notebook từ đầu khoảng 45 phút.
