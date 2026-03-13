# 🎓 Student Academic Status Prediction

Dự đoán tình trạng học vụ sinh viên (Bình thường / Cảnh báo / Thôi học) từ dữ liệu điểm danh, tài chính và văn bản tiếng Việt.

**Leaderboard Score:** `0.87983` Macro F1 &nbsp;|&nbsp; **Baseline:** `0.86700` &nbsp;|&nbsp; **Cải thiện:** `+1.283 điểm (+1.5%)`

---

## 📖 Mô tả bài toán

| Thông số   | Chi tiết                             |
| ---------- | ------------------------------------ |
| **Task**   | Multi-class Classification (3 lớp)   |
| **Metric** | Macro F1-Score                       |
| **Train**  | 6,000 mẫu × 54 cột                   |
| **Test**   | 4,000 mẫu × 53 cột                   |
| **Nhãn 0** | Bình thường (Normal) — 64.3%         |
| **Nhãn 1** | Cảnh báo học vụ (Warning) — 22.3%    |
| **Nhãn 2** | Thôi học / Bị đuổi (Dropout) — 13.4% |

---

## 📁 Cấu trúc dự án

```
project/
│
├── Final.ipynb                       # Entry point chính — chạy file này
│
├── fe/
│   ├── upgrade_part1_cleaning.py     # Bước 1: Làm sạch dữ liệu
│   └── upgrade_part2_features.py     # Bước 2: Feature engineering + PhoBERT
│
├── data/
│   ├── train.csv                     # Dữ liệu huấn luyện (6,000 mẫu)
│   └── test.csv                      # Dữ liệu dự đoán (4,000 mẫu)
│
├── requirements.txt
├── README.md
├── HUONG_DAN_THUC_THI.md
└── GIAI_THICH_REPRODUCIBILITY.md
```

**File sinh ra sau khi chạy:**

```
data/train_clean_v2.csv               # Output bước 1
data/test_clean_v2.csv                # Output bước 1
data/train_fe_v2.csv                  # Output bước 2 (~136 features)
data/test_fe_v2.csv                   # Output bước 2
submission_FINAL_ELITE_TEST_088.csv   # Kết quả cuối nộp
```

---

## 🏗️ Kiến trúc Pipeline

```
train.csv ──┐
            ├──► Part1: Cleaning ──► Part2: Feature Engineering (PhoBERT)
test.csv  ──┘        ~15s                       ~95s (GPU)
                                           ~136 features
                                                │
                              ┌─────────────────┴────────────────┐
                              ▼                                   ▼
                   CatBoost 5-Fold                       XGBoost Full-train
                   random_state=10                       random_state=42
                   class_weights=[1.0, 1.8, 2.5]        n_estimators=1000
                   trọng số = 70%                        trọng số = 30%
                              │                                   │
                              └─────────────┬────────────────────┘
                                            ▼
                                  Blend 70/30
                                  + Dropout Boost ×1.15
                                            │
                                       base_preds
                                            │
                                  Elite CatBoost 5-Fold
                                  class_weights={0:0.9, 1:1.8, 2:3.8}
                                  Lọc confidence > 80%
                                            │
                                            ▼
                             submission_FINAL_ELITE_TEST_088.csv
                             {0: 2603, 1: 824, 2: 573}
```

---

## 📊 Kết quả

| Phiên      | Leaderboard F1 |
| ---------- | -------------- |
| Baseline   | 0.86700        |
| Ngày 14/02 | ~0.871         |
| Ngày 15/02 | ~0.873         |
| Ngày 19/02 | ~0.87795       |
| **Final**  | **0.87983**    |

-- ĐỂ ĐẠT ĐƯỢC KẾT QUẢ CÓ THỂ BỎ QUA VIỆC THỰC THI hai file fe sử dụng data/train_fe_v2.csv và data/test_fe_v2.csv
**Phân phối nhãn dự đoán cuối:**

| Nhãn        | Train (thực)  | Test (dự đoán) |
| ----------- | ------------- | -------------- |
| 0 – Normal  | 3858 (64.30%) | 2603 (65.08%)  |
| 1 – Warning | 1340 (22.33%) | 824 (20.60%)   |
| 2 – Dropout | 802 (13.37%)  | 573 (14.33%)   |

---