"""
===================================================================================
STUDENT STATUS PREDICTION - UPGRADED VERSION
===================================================================================
Baseline: 0.867 Macro F1
Target: 0.88-0.90+ Macro F1

IMPROVEMENTS:
1. ✅ Extended teencode dictionary (50+ patterns)
2. ✅ More aggressive outlier handling
3. ✅ Better categorical encoding
4. ✅ Text length features
5. ✅ Enhanced attendance features

Author: Upgraded by Claude (Kaggle Grandmaster)
===================================================================================
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PART 1: ENHANCED DATA CLEANING")
print("="*80)

# ==========================================
# 1. EXTENDED TEENCODE DICTIONARY (50+ patterns)
# ==========================================
teencode_dict = {
    # Original patterns
    'mik': 'mình', 'mk': 'mình', 'mjk': 'mình', 'mjnh': 'mình',
    'ik': 'đi',
    'hok': 'không', 'ko': 'không', 'k': 'không', 'hem': 'không',
    'dc': 'được', 'đc': 'được', 'duoc': 'được',
    'bit': 'biết', 'bik': 'biết', 'bt': 'biết',
    'j': 'gì', 'gj': 'gì', 'ji': 'gì',
    'vs': 'với', 'voi': 'với', 'voii': 'với',
    'h': 'giờ', 'r': 'rồi', 'roj': 'rồi', 'ròi': 'rồi',
    
    # Extended patterns (NEW!)
    'thik': 'thích', 'thjk': 'thích', 'thix': 'thích',
    'cx': 'cũng', 'cug': 'cũng', 'cugx': 'cũng',
    'nhiu': 'nhiều', 'nhju': 'nhiều', 'nhìu': 'nhiều',
    'lm': 'làm', 'lam': 'làm', 'lamf': 'làm',
    'nv': 'như vậy', 'ntn': 'như thế nào',
    'trog': 'trong', 'trongg': 'trong',
    'ng': 'người', 'ngj': 'người', 'nguoi': 'người',
    'hjc': 'hiểu', 'hju': 'hiểu', 'hjk': 'hiểu',
    'thj': 'thì', 'thi': 'thì', 'thif': 'thì',
    'nx': 'nữa', 'nưa': 'nữa', 'nửa': 'nữa',
    'bh': 'bao giờ', 'bjo': 'bao giờ',
    'ms': 'mới', 'moi': 'mới', 'mjoi': 'mới',
    'chs': 'chưa', 'ch': 'chưa', 'chwa': 'chưa',
    'nhma': 'nhưng mà', 'nma': 'nhưng mà',
    'fai': 'phải', 'pải': 'phải',
    'đag': 'đang', 'dang': 'đang', 'dangf': 'đang',
    'vx': 'vậy', 'z': 'vậy', 'zậy': 'vậy',
    
    # Student-specific
    'sv': 'sinh viên', 'gv': 'giáo viên', 'hs': 'học sinh',
    'th': 'thực hành', 'lt': 'lý thuyết',
    'đh': 'đại học', 'cd': 'cao đẳng',
    'mon': 'môn', 'hoc': 'học', 'ky': 'kỳ',
    'thi': 'thi', 'kt': 'kiểm tra',
    
    # Sentiment words (NEW!)
    'tot': 'tốt', 'totj': 'tốt', 'good': 'tốt',
    'xau': 'xấu', 'te': 'tệ', 'bad': 'xấu',
    'kho': 'khó', 'khoq': 'khó',
    'de': 'dễ', 'dex': 'dễ', 'easy': 'dễ',
    'muon': 'muộn', 'late': 'muộn',
    'som': 'sớm', 'early': 'sớm',
    'vang': 'vắng', 'absent': 'vắng',
    'qá': 'quá', 'wa': 'quá', 'wá': 'quá',
    'chac': 'chắc', 'chak': 'chắc',
}

# ==========================================
# 2. ENHANCED CLEANING FUNCTION
# ==========================================
def clean_vietnamese_text(text):
    """Enhanced cleaning with better regex and normalization"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    
    # Fix common typos BEFORE teencode
    text = text.replace('tuyểển', 'tuyển')
    text = text.replace('hjện', 'hiện')
    text = text.replace('hjọc', 'học')
    
    # Fix Teencode with word boundaries
    for code, real in teencode_dict.items():
        pattern = r'\b' + re.escape(code) + r'\b'
        text = re.sub(pattern, real, text)
    
    # Keep Vietnamese characters + numbers + spaces
    text = re.sub(r'[^\w\s\dáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==========================================
# 3. LOAD & PREPARE DATA
# ==========================================
print("\n[1/7] Loading Data...")
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_train['dataset_source'] = 'train'
df_test['dataset_source'] = 'test'
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

print(f"Train: {df_train.shape}, Test: {df_test.shape}")

# ==========================================
# 4. CLEAN CATEGORIES
# ==========================================
print("\n[2/7] Cleaning Categorical Data...")

# Admission_Mode
df_all['Admission_Mode'] = df_all['Admission_Mode'].apply(clean_vietnamese_text)

# English_Level - Enhanced mapping
df_all['English_Level_Clean'] = df_all['English_Level'].astype(str).str.lower().str.strip().str.rstrip('.')

english_map = {
    'a0': 0, 'a1': 1, 'a2': 2,
    'b1': 3, 'b2': 4,
    'c1': 5, 'c2': 6,
    'ielts 6.0+': 5, 'ielts 60+': 5, 'ielts 6+': 5,
    'ielts 7.0+': 6, 'ielts 7+': 6,
}

df_all['English_Level_Mapped'] = df_all['English_Level_Clean'].map(english_map)
mode_val = df_all['English_Level_Mapped'].mode()[0]
df_all['English_Level_Mapped'] = df_all['English_Level_Mapped'].fillna(mode_val).astype(int)

# ==========================================
# 5. CLEAN TEXT (NEW: Extract features BEFORE cleaning)
# ==========================================
print("\n[3/7] Processing Text Data...")

text_cols = ['Personal_Essay', 'Advisor_Notes']

for col in text_cols:
    # TEXT LENGTH FEATURES (NEW!)
    df_all[f'{col}_length'] = df_all[col].fillna('').astype(str).str.len()
    df_all[f'{col}_word_count'] = df_all[col].fillna('').astype(str).str.split().str.len()
    
    # Clean text
    df_all[col] = df_all[col].apply(clean_vietnamese_text)
    
    # SENTIMENT FEATURES (NEW!)
    # Count positive/negative words
    positive_words = ['tốt', 'giỏi', 'xuất sắc', 'tuyệt', 'thích', 'yêu', 'vui']
    negative_words = ['không', 'xấu', 'tệ', 'kém', 'bỏ', 'chán', 'khó', 'muộn', 'vắng']
    
    df_all[f'{col}_positive_count'] = df_all[col].fillna('').apply(
        lambda x: sum(word in x for word in positive_words)
    )
    df_all[f'{col}_negative_count'] = df_all[col].fillna('').apply(
        lambda x: sum(word in x for word in negative_words)
    )
    df_all[f'{col}_sentiment_ratio'] = (
        df_all[f'{col}_positive_count'] / (df_all[f'{col}_negative_count'] + 1)
    )

print("  ✓ Created text length & sentiment features")

# ==========================================
# 6. ENHANCED NUMERICAL CLEANING
# ==========================================
print("\n[4/7] Cleaning Numerical Data...")

# Tuition_Debt
df_all['Tuition_Debt'] = pd.to_numeric(df_all['Tuition_Debt'], errors='coerce').fillna(0)

# Attendance - MORE AGGRESSIVE OUTLIER HANDLING (NEW!)
att_cols = [c for c in df_all.columns if 'Att_Subject_' in c]

for col in att_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    # Enhanced outlier logic
    df_all.loc[df_all[col] < 0, col] = np.nan      # Negative → NaN
    df_all.loc[df_all[col] > 20, col] = np.nan     # > 20 → NaN
    df_all.loc[df_all[col] == 0, col] = 0.5        # NEW: 0 → 0.5 (distinguish from NaN)

# Count_F
df_all['Count_F'] = pd.to_numeric(df_all['Count_F'], errors='coerce').fillna(0)

# ==========================================
# 7. ENCODING
# ==========================================
print("\n[5/7] Encoding Categorical Variables...")

other_cats = ['Gender', 'Hometown', 'Current_Address', 'Club_Member', 'Admission_Mode']
le = LabelEncoder()

for col in other_cats:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str)
        df_all[f'{col}_Encoded'] = le.fit_transform(df_all[col])

# ==========================================
# 8. EXPORT CLEANED DATA
# ==========================================
print("\n[6/7] Exporting Cleaned Data...")

train_clean = df_all[df_all['dataset_source'] == 'train'].drop(columns=['dataset_source'])
test_clean = df_all[df_all['dataset_source'] == 'test'].drop(columns=['dataset_source', 'Academic_Status'])

train_clean.to_csv('data/train_clean_v2.csv', index=False)
test_clean.to_csv('data/test_clean_v2.csv', index=False)

print("✅ Enhanced Cleaning Complete!")
print(f"   - Extended teencode: {len(teencode_dict)} patterns")
print(f"   - Text features: {len([c for c in df_all.columns if 'length' in c or 'count' in c or 'sentiment' in c])} new features")
print(f"   - Files saved: train_clean_v2.csv, test_clean_v2.csv")
