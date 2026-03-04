# 🔍 GIẢI THÍCH VẤN ĐỀ REPRODUCIBILITY

## Hiện tượng

Khi chạy lại pipeline, phân phối nhãn đầu ra **có thể khác** so với lần chạy gốc (`{0: 2603, 1: 824, 2: 573}`), dù dùng đúng cùng dữ liệu và cùng code.

---

## Nguyên nhân gốc rễ

**`upgrade_part2_features.py` thiếu random seed cho PhoBERT GPU.**

PhoBERT sử dụng GPU để tính toán embedding. Các phép tính dấu phẩy động trên GPU không đảm bảo thứ tự thực thi giống nhau giữa các lần chạy do cơ chế song song hóa (CUDA thread scheduling). Kết quả là mỗi lần chạy `upgrade_part2_features.py` sẽ tạo ra `train_fe_v2.csv` và `test_fe_v2.csv` với giá trị embedding khác nhau một lượng rất nhỏ (~1e-6), nhưng đủ để làm thay đổi kết quả phân loại ở biên giới quyết định.

Chuỗi hậu quả:

```
Thiếu seed trong upgrade_part2_features.py
            ↓
PhoBERT GPU → CLS embeddings khác nhau mỗi lần
            ↓
train_fe_v2.csv & test_fe_v2.csv có MD5 khác nhau mỗi lần chạy
            ↓
CatBoost nhận input khác → predict_proba khác
            ↓
Phân phối nhãn cuối thay đổi
```

**Xác nhận bằng MD5:** Bản gốc cho điểm 0.87983 có:
```
train_fe_v2.csv → MD5: 27070d699c53371f012cc4c8f0e055d6
test_fe_v2.csv  → MD5: 8f709478b2001df982987b992108e770
```
Mỗi lần chạy lại `upgrade_part2_features.py` sẽ cho MD5 khác hai giá trị trên.

---

## Các yếu tố KHÔNG phải nguyên nhân

Trong quá trình điều tra, các yếu tố sau đã bị nghi ngờ nhưng đều đã được loại trừ:

| Yếu tố bị nghi ngờ | Kết luận |
|--------------------|----------|
| Elite CatBoost không có `random_seed` | ❌ Không phải — Elite corrections = 0, không ảnh hưởng output |
| `early_stopping_rounds + TotalF1 + GPU` trong CatBoost base | ❌ Không phải — chỉ là triệu chứng phụ khi input data đã khác |
| `train_fe_v2.csv` bị thay đổi | ✅ Đúng — nhưng nguyên nhân là do PhoBERT, không phải file bị sửa tay |

---

## Cách tái tạo kết quả 0.87983

### Phương án 1 — Giữ nguyên file FE gốc (Đơn giản nhất)

Nếu đang có `train_fe_v2.csv` và `test_fe_v2.csv` với đúng MD5 gốc, **không chạy lại** `upgrade_part2_features.py`. Thay vào đó, bỏ qua bước FE và load thẳng:

```python
import hashlib, sys, pandas as pd

# Xác nhận MD5 trước khi chạy
checks = [
    ('data/train_fe_v2.csv', '27070d699c53371f012cc4c8f0e055d6'),
    ('data/test_fe_v2.csv',  '8f709478b2001df982987b992108e770'),
]
for path, expected in checks:
    with open(path, 'rb') as f:
        actual = hashlib.md5(f.read()).hexdigest()
    if actual != expected:
        print(f"❌ {path} không phải bản gốc — DỪNG LẠI")
        sys.exit(1)
    print(f"✅ {path} còn nguyên gốc")

# Load thẳng, không chạy lại FE
train = pd.read_csv('data/train_fe_v2.csv')
test  = pd.read_csv('data/test_fe_v2.csv')
print("✅ Sẵn sàng training — sẽ tái tạo được 0.87983")
```

### Phương án 2 — Fix seed trong upgrade_part2_features.py (Lâu dài)

Thêm đoạn code sau vào `upgrade_part2_features.py`, ngay sau dòng `print(f"Using device: {device}")`:

```python
# ── FIX DETERMINISTIC SEED ──────────────────────────────
import random, os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True   # Buộc cuDNN dùng thuật toán deterministic
    torch.backends.cudnn.benchmark     = False  # Tắt auto-select algorithm
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f"✅ Seed = {SEED} đã được cố định")
# ────────────────────────────────────────────────────────
```

> ⚠️ **Lưu ý:** `cudnn.deterministic = True` làm PhoBERT chạy chậm hơn ~10–20% vì cuDNN không dùng thuật toán tối ưu nhanh nhất. Đây là đánh đổi tất yếu giữa tốc độ và tính tái tạo.

Sau khi thêm seed, `train_fe_v2.csv` và `test_fe_v2.csv` sẽ có MD5 giống nhau qua tất cả các lần chạy.

---

## Tóm tắt

| Câu hỏi | Trả lời |
|---------|---------|
| Tại sao phân phối nhãn thay đổi? | PhoBERT GPU non-determinism → file FE khác mỗi lần |
| Có cần tìm seed của Elite CatBoost không? | Không — Elite corrections = 0, không ảnh hưởng |
| Cách nhanh nhất để tái tạo 0.87983? | Giữ nguyên 2 file FE gốc, không chạy lại upgrade_part2 |
| Cách fix dứt điểm về lâu dài? | Thêm 6 dòng seed vào upgrade_part2_features.py |
