import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

MATRIX = "04_ML/GSE48350_RMA_ML_matrix.csv"
META = "03_Preprocessing/GSE48350_RMA_metadata_aligned.csv"

TRAIN_META = "04_ML/GSE48350_RMA_train_meta.csv"
TEST_META = "04_ML/GSE48350_RMA_test_meta.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X = pd.read_csv(MATRIX, index_col=0)
meta = pd.read_csv(META)

# ------------------------------------------------------------
# Verify sample identity
# ------------------------------------------------------------

if set(X.index) != set(meta["GSM"]):
    raise ValueError("Expression samples and metadata samples do not match.")

meta = meta.set_index("GSM").loc[X.index]

# ------------------------------------------------------------
# Diagnosis
# ------------------------------------------------------------

y = meta["Diagnosis"]

print("=" * 60)
print("GSE48350 RMA TRAIN/TEST SPLIT")
print("=" * 60)

print("\nTotal samples:", len(meta))

print("\nFull dataset diagnosis:")
print(y.value_counts())

# ------------------------------------------------------------
# Stratified split
# ------------------------------------------------------------

train_ids, test_ids = train_test_split(
    meta.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

train_meta = meta.loc[train_ids].copy()
test_meta = meta.loc[test_ids].copy()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

train_meta.reset_index().to_csv(
    TRAIN_META,
    index=False
)

test_meta.reset_index().to_csv(
    TEST_META,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\nTraining samples:", len(train_meta))
print("Test samples:", len(test_meta))

print("\nTraining diagnosis:")
print(train_meta["Diagnosis"].value_counts())

print("\nTest diagnosis:")
print(test_meta["Diagnosis"].value_counts())

print("\nTraining IDs:")
print(train_meta.index.tolist())

print("\nTest IDs:")
print(test_meta.index.tolist())

print("\nSaved:")
print(TRAIN_META)
print(TEST_META)

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)
