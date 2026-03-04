# 📘 HƯỚNG DẪN THỰC THI CHI TIẾT

---

## 1. YÊU CẦU TRƯỚC KHI CHẠY

### Cấu trúc thư mục cần có

```
project/
├── Final.ipynb
├── fe/
│   ├── upgrade_part1_cleaning.py
│   └── upgrade_part2_features.py
└── data/
    ├── train.csv
    └── test.csv
```

### Yêu cầu phần cứng

|      | Tối thiểu      | Khuyến nghị          |
| ---- | -------------- | -------------------- |
| RAM  | 8 GB           | 16 GB                |
| GPU  | Không bắt buộc | T4 / RTX (4GB+ VRAM) |
| Disk | 5 GB           | 10 GB                |

---

## 2. CHẠY TRÊN GOOGLE COLAB (Khuyến nghị)

### Bước 1 — Bật GPU T4

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

### Bước 2 — Tạo thư mục và upload file

```python
import os
os.makedirs('fe', exist_ok=True)
os.makedirs('data', exist_ok=True)
print("✅ Đã tạo thư mục")
```

Upload từng file đúng vị trí:

| File cần upload             | Đích               |
| --------------------------- | ------------------ |
| `upgrade_part1_cleaning.py` | `fe/`              |
| `upgrade_part2_features.py` | `fe/`              |
| `train.csv`                 | `data/`            |
| `test.csv`                  | `data/`            |
| `Final.ipynb`               | `/content/` (root) |

Di chuyển file sau khi upload:

```python
import shutil, os

moves = {
    'upgrade_part1_cleaning.py': 'fe/upgrade_part1_cleaning.py',
    'upgrade_part2_features.py': 'fe/upgrade_part2_features.py',
    'train.csv':                 'data/train.csv',
    'test.csv':                  'data/test.csv',
}
for src, dst in moves.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"✅ Đã di chuyển: {src} → {dst}")
```

### Bước 3 — Cài thư viện

```python
!pip install -q catboost xgboost transformers underthesea accelerate tqdm
```

> ⏱️ Thời gian: ~2–3 phút

### Bước 4 — Chạy Final.ipynb

Mở `Final.ipynb` và **Run All**. Pipeline tự động chạy từ A–Z:

| Bước     | Nội dung                            | Thời gian (T4 GPU) |
| -------- | ----------------------------------- | ------------------ |
| 1        | upgrade_part1_cleaning.py           | ~15 giây           |
| 2        | upgrade_part2_features.py (PhoBERT) | ~95 giây           |
| 3        | CatBoost 5-Fold (70% weight)        | ~3–5 phút          |
| 4        | XGBoost Full-train (30% weight)     | ~1–2 phút          |
| 5        | Elite CatBoost 5-Fold               | ~5–8 phút          |
| **Tổng** |                                     | **~12–18 phút**    |

Output mong đợi cuối pipeline:

```
✅ Hoàn thành fe/upgrade_part1_cleaning.py trong ~15s
✅ Hoàn thành fe/upgrade_part2_features.py trong ~95s
📉 Phân bổ TRƯỚC KHI bơm: {0: 2603, 1: 852, 2: 545}
📈 Phân bổ SAU KHI bơm:   {0: 2603, 1: 824, 2: 573}
👉 Elite corrections: 0
📊 Phân bổ nhãn cuối: {0: 2603, 1: 824, 2: 573}
```

### Bước 5 — Tải kết quả

```python
from google.colab import files
files.download('submission_FINAL_ELITE_TEST_088.csv')
```

---

## 3. CHẠY LOCAL (Python)

### Cài đặt

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Thực thi từng bước

```bash
# Bước 1
python fe/upgrade_part1_cleaning.py
# → data/train_clean_v2.csv, data/test_clean_v2.csv

# Bước 2
python fe/upgrade_part2_features.py
# → data/train_fe_v2.csv, data/test_fe_v2.csv

# Bước 3: Mở và chạy Final.ipynb
jupyter notebook Final.ipynb
```

### Chạy không có GPU

Đổi `task_type="GPU"` → `task_type="CPU"` và `device='cuda'` → `device='cpu'` trong `Final.ipynb`.

> ⏱️ Thời gian ước tính khi dùng CPU: ~45–90 phút (PhoBERT chiếm phần lớn)

---

## 4. KIỂM TRA KẾT QUẢ

```python
import pandas as pd

result = pd.read_csv('submission_FINAL_ELITE_TEST_088.csv')
print(f"Tổng mẫu: {len(result)}")
print(result['Academic_Status'].value_counts().sort_index())

# Kết quả mong đợi:
# 0    2603
# 1     824
# 2     573
```

---

## 5. XỬ LÝ LỖI THƯỜNG GẶP

**`FileNotFoundError: data/train.csv`**

```python
import os
print(os.getcwd())        # Kiểm tra thư mục hiện tại
os.chdir('/content')      # Nếu sai, đổi về /content (Colab)
```

**`ModuleNotFoundError: No module named 'underthesea'`**

```bash
pip install underthesea
```

**`CUDA out of memory`**

```python
# Giảm batch_size trong upgrade_part2_features.py
# Tìm dòng: get_bert_embeddings(text, batch_size=32)
# Đổi thành:                   batch_size=16
```

**`CatBoostError: GPU not found`**

```python
# Đổi task_type="GPU" → task_type="CPU" trong Final.ipynb
```

**Phân phối nhãn khác kết quả mong đợi**

Xem `DOCUMENT/REPRODUCIBILITY.md` để hiểu nguyên nhân và cách xử lý.
