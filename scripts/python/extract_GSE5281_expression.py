import gzip
import pandas as pd

geo_file = "01_GEO_Data/GSE5281/GSE5281_series_matrix.txt.gz"
metadata_file = "02_Metadata/GSE5281_hippocampus_metadata.csv"
output_file = "03_Preprocessing/GSE5281_hippocampus_expression.csv"

# --------------------------------------------------
# 1. Read validation sample metadata
# --------------------------------------------------

metadata = pd.read_csv(metadata_file)
target_gsms = metadata["GSM"].astype(str).tolist()

print("Target validation samples:", len(target_gsms))

# --------------------------------------------------
# 2. Locate and read the GEO expression table
# --------------------------------------------------

table_rows = []
reading_table = False

with gzip.open(geo_file, "rt", encoding="utf-8", errors="replace") as f:

    for line in f:

        line = line.rstrip("\n")

        if line.startswith("!series_matrix_table_begin"):
            reading_table = True
            continue

        if line.startswith("!series_matrix_table_end"):
            break

        if reading_table:
            table_rows.append(line.split("\t"))

# --------------------------------------------------
# 3. Convert to DataFrame
# --------------------------------------------------

header = [x.strip('"') for x in table_rows[0]]

data = []

for row in table_rows[1:]:
    row = [x.strip('"') for x in row]

    if len(row) == len(header):
        data.append(row)

expression = pd.DataFrame(data, columns=header)

print("\nFull expression matrix:")
print("Rows (probes):", expression.shape[0])
print("Columns:", expression.shape[1])

# --------------------------------------------------
# 4. Check required GSMs
# --------------------------------------------------

missing = [gsm for gsm in target_gsms if gsm not in expression.columns]

if missing:
    print("\nERROR: Required GSMs not found:")
    print(missing)
    raise SystemExit(1)

# --------------------------------------------------
# 5. Extract exactly the 23 hippocampus samples
# --------------------------------------------------

probe_col = "ID_REF"

expression = expression[[probe_col] + target_gsms]

# --------------------------------------------------
# 6. Convert expression values to numeric
# --------------------------------------------------

for gsm in target_gsms:
    expression[gsm] = pd.to_numeric(
        expression[gsm],
        errors="coerce"
    )

# --------------------------------------------------
# 7. Basic QC
# --------------------------------------------------

print("\nExtracted hippocampus matrix:")
print("Probes:", expression.shape[0])
print("Samples:", len(target_gsms))
print("Matrix shape:", expression.shape)

print("\nMissing values:")
print(expression[target_gsms].isna().sum().sum())

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

expression.to_csv(output_file, index=False)

print("\nSaved:")
print(output_file)
