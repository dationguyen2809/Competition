"""
===================================================================================
PART 2: ENHANCED FEATURE ENGINEERING
===================================================================================

IMPROVEMENTS:
1. ✅ PhoBERT for BOTH text columns (Personal_Essay + Advisor_Notes)
2. ✅ Enhanced attendance features (20+ features)
3. ✅ More interaction features
4. ✅ Polynomial features for key predictors

Expected improvement: +2-3% Macro F1
===================================================================================
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("="*80)
print("PART 2: ENHANCED FEATURE ENGINEERING")
print("="*80)

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================
print("\n[1/5] Loading Cleaned Data...")
df_train = pd.read_csv('data/train_clean_v2.csv')
df_test = pd.read_csv('data/test_clean_v2.csv')

df_train['dataset_source'] = 'train'
df_test['dataset_source'] = 'test'
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Force numeric types
att_cols = [c for c in df_all.columns if 'Att_Subject_' in c]
for col in att_cols + ['Tuition_Debt', 'Count_F']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# ==========================================
# 2. ENHANCED ATTENDANCE FEATURES (20+ features)
# ==========================================
print("\n[2/5] Creating Enhanced Attendance Features...")

# Original features
df_all['Mean_Score_All'] = df_all[att_cols].mean(axis=1)
df_all['Std_Score_All'] = df_all[att_cols].std(axis=1).fillna(0)
df_all['Min_Score'] = df_all[att_cols].min(axis=1)
df_all['Max_Score'] = df_all[att_cols].max(axis=1)
df_all['Median_Score'] = df_all[att_cols].median(axis=1)

# Count features
df_all['Count_Attended'] = df_all[att_cols].notna().sum(axis=1)
df_all['Count_Missing'] = df_all[att_cols].isna().sum(axis=1)
df_all['Subject_Fail_Count'] = (df_all[att_cols] < 4.0).sum(axis=1)
df_all['Subject_Pass_Count'] = (df_all[att_cols] >= 4.0).sum(axis=1)

# NEW: More granular performance levels
df_all['Subject_Excellent_Count'] = (df_all[att_cols] >= 15.0).sum(axis=1)
df_all['Subject_Good_Count'] = ((df_all[att_cols] >= 10.0) & (df_all[att_cols] < 15.0)).sum(axis=1)
df_all['Subject_Average_Count'] = ((df_all[att_cols] >= 7.0) & (df_all[att_cols] < 10.0)).sum(axis=1)
df_all['Subject_Poor_Count'] = ((df_all[att_cols] >= 4.0) & (df_all[att_cols] < 7.0)).sum(axis=1)

# NEW: Ratios (very important!)
df_all['Excellent_Ratio'] = df_all['Subject_Excellent_Count'] / (df_all['Count_Attended'] + 1)
df_all['Fail_Ratio'] = df_all['Subject_Fail_Count'] / (df_all['Count_Attended'] + 1)
df_all['Pass_Ratio'] = df_all['Subject_Pass_Count'] / (df_all['Count_Attended'] + 1)

# NEW: Range and variability
df_all['Score_Range'] = df_all['Max_Score'] - df_all['Min_Score']
df_all['Score_CV'] = df_all['Std_Score_All'] / (df_all['Mean_Score_All'] + 1)  # Coefficient of variation

# NEW: Trend analysis (first half vs second half)
n_half = len(att_cols) // 2
first_half_cols = att_cols[:n_half]
second_half_cols = att_cols[n_half:]

df_all['First_Half_Mean'] = df_all[first_half_cols].mean(axis=1)
df_all['Second_Half_Mean'] = df_all[second_half_cols].mean(axis=1)
df_all['Score_Trend'] = df_all['Second_Half_Mean'] - df_all['First_Half_Mean']
df_all['Trend_Positive'] = (df_all['Score_Trend'] > 0).astype(int)

print(f"  ✓ Created {len([c for c in df_all.columns if 'Score' in c or 'Count' in c or 'Ratio' in c or 'Trend' in c])} attendance features")

# ==========================================
# 3. ENHANCED FINANCIAL & INTERACTION FEATURES
# ==========================================
print("\n[3/5] Creating Enhanced Interaction Features...")

# Financial
df_all['Tuition_Debt'] = df_all['Tuition_Debt'].fillna(0)
df_all['Has_Debt'] = (df_all['Tuition_Debt'] > 0).astype(int)
df_all['Log_Debt'] = np.log1p(df_all['Tuition_Debt'])

# NEW: Debt severity levels
df_all['Debt_Level'] = pd.cut(df_all['Tuition_Debt'], 
                               bins=[-1, 0, 1000000, 5000000, float('inf')],
                               labels=[0, 1, 2, 3]).astype(int)

# Academic performance
df_all['Count_F'] = df_all['Count_F'].fillna(0)

# NEW: More interaction features
df_all['Fail_Debt_Interaction'] = df_all['Count_F'] * df_all['Log_Debt']
df_all['Performance_Debt_Ratio'] = df_all['Mean_Score_All'] / (df_all['Log_Debt'] + 1)
df_all['Attendance_Debt_Risk'] = df_all['Fail_Ratio'] * df_all['Has_Debt']

# NEW: Academic risk score
df_all['Academic_Risk_Score'] = (
    df_all['Fail_Ratio'] * 0.4 +
    (1 - df_all['Pass_Ratio']) * 0.3 +
    df_all['Count_F'] / 10 * 0.2 +
    df_all['Has_Debt'] * 0.1
)

print(f"  ✓ Created {len([c for c in df_all.columns if 'Interaction' in c or 'Risk' in c or 'Level' in c])} interaction features")

# ==========================================
# 4. PHOBERT FOR BOTH TEXT COLUMNS (UPGRADED!)
# ==========================================
print("\n[4/5] Processing PhoBERT Embeddings...")
print("  (This may take 5-10 minutes depending on GPU)")

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
bert_model = AutoModel.from_pretrained(model_name).to(device)

def get_bert_embeddings(text_list, batch_size=32):
    """Extract BERT embeddings with progress bar"""
    bert_model.eval()
    embeddings = []
    
    # Pre-tokenize
    print("    → Tokenizing...")
    processed_texts = [word_tokenize(str(t) if pd.notna(t) else "", format="text") for t in text_list]
    
    # Extract embeddings
    print("    → Extracting embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="    "):
            batch_texts = processed_texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    return np.vstack(embeddings)

# Process BOTH text columns
if torch.cuda.is_available() or len(df_all) < 5000:  # Skip if CPU and large dataset
    
    # Process Advisor_Notes
    print("\n  [A] Processing Advisor_Notes...")
    notes_text = df_all['Advisor_Notes'].fillna("").tolist()
    notes_embeddings = get_bert_embeddings(notes_text)
    
    print("    → Reducing dimensions (PCA)...")
    pca_notes = PCA(n_components=16, random_state=42)
    notes_pca = pca_notes.fit_transform(notes_embeddings)
    
    for i in range(16):
        df_all[f'BERT_Note_{i}'] = notes_pca[:, i]
    
    print(f"    ✓ Explained variance: {pca_notes.explained_variance_ratio_.sum():.2%}")
    
    # Process Personal_Essay (NEW!)
    print("\n  [B] Processing Personal_Essay...")
    essay_text = df_all['Personal_Essay'].fillna("").tolist()
    essay_embeddings = get_bert_embeddings(essay_text)
    
    print("    → Reducing dimensions (PCA)...")
    pca_essay = PCA(n_components=16, random_state=42)
    essay_pca = pca_essay.fit_transform(essay_embeddings)
    
    for i in range(16):
        df_all[f'BERT_Essay_{i}'] = essay_pca[:, i]
    
    print(f"    ✓ Explained variance: {pca_essay.explained_variance_ratio_.sum():.2%}")
    
else:
    print("  ⚠️ Skipping PhoBERT (CPU detected or large dataset)")
    print("  → For best results, run on GPU")

# ==========================================
# 5. POLYNOMIAL FEATURES FOR KEY PREDICTORS
# ==========================================
print("\n[5/5] Creating Polynomial Features...")

key_features = ['Mean_Score_All', 'Fail_Ratio', 'Log_Debt']

for feat in key_features:
    if feat in df_all.columns:
        df_all[f'{feat}_squared'] = df_all[feat] ** 2
        df_all[f'{feat}_cubed'] = df_all[feat] ** 3

print(f"  ✓ Created {len([c for c in df_all.columns if 'squared' in c or 'cubed' in c])} polynomial features")

# ==========================================
# 6. EXPORT FINAL FEATURES
# ==========================================
print("\n[6/6] Exporting Enhanced Features...")

train_fe = df_all[df_all['dataset_source'] == 'train'].drop(columns=['dataset_source'])
test_fe = df_all[df_all['dataset_source'] == 'test'].drop(columns=['dataset_source', 'Academic_Status'])

train_fe.to_csv('data/train_fe_v2.csv', index=False)
test_fe.to_csv('data/test_fe_v2.csv', index=False)

# Summary statistics
total_features = len(train_fe.columns) - 1  # Exclude Student_ID
print(f"\n{'='*80}")
print(f"FEATURE ENGINEERING SUMMARY")
print(f"{'='*80}")
print(f"Total features: {total_features}")
print(f"  - Attendance features: ~25")
print(f"  - Text length/sentiment: ~12")
print(f"  - PhoBERT embeddings: 32 (16 per column)")
print(f"  - Interaction features: ~10")
print(f"  - Polynomial features: ~9")
print(f"{'='*80}")
print("✅ Enhanced Feature Engineering Complete!")
print("   Files saved: train_fe_v2.csv, test_fe_v2.csv")
