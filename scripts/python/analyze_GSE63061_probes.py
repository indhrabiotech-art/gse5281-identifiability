import gzip
import csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

FILE = "09_CrossTissue/GSE63061/GSE63061_normalized.txt.gz"

PROBES = {
    "ABCA6_probe1": "ILMN_1701551",
    "ABCA6_probe2": "ILMN_1795507",
    "CRLF1": "ILMN_1681515",
    "TNFRSF11B": "ILMN_1676663",
}

found = {}

inside = False

with gzip.open(FILE, "rt") as fh:

    reader = csv.reader(fh, delimiter="\t")

    for row in reader:

        if not row:
            continue

        if row[0] == "!series_matrix_table_begin":
            inside = True
            continue

        if row[0] == "!series_matrix_table_end":
            break

        if not inside:
            continue

        probe = row[0].strip("\"").strip('"')

        for name, probe_id in PROBES.items():

            if probe == probe_id:

                values = []

                for v in row[1:]:
                    try:
                        values.append(float(v.strip('"')))
                    except ValueError:
                        values.append(np.nan)

                found[name] = np.array(values, dtype=float)


print("=" * 70)
print("GSE63061 THREE-GENE PROBE ANALYSIS")
print("=" * 70)

print("\nProbe presence:")

for name, probe in PROBES.items():

    if name in found:
        print(f"{name:20s} {probe:15s} FOUND")
    else:
        print(f"{name:20s} {probe:15s} NOT FOUND")


print("\n" + "=" * 70)
print("EXPRESSION SUMMARY")
print("=" * 70)

summary = []

for name, values in found.items():

    valid = values[np.isfinite(values)]

    summary.append({
        "Probe": name,
        "N": len(valid),
        "Mean": np.mean(valid),
        "SD": np.std(valid, ddof=1),
        "Median": np.median(valid),
        "Min": np.min(valid),
        "Max": np.max(valid)
    })

summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("ABCA6 PROBE CONCORDANCE")
print("=" * 70)

a = found["ABCA6_probe1"]
b = found["ABCA6_probe2"]

mask = np.isfinite(a) & np.isfinite(b)

pearson_r, pearson_p = pearsonr(
    a[mask],
    b[mask]
)

spearman_rho, spearman_p = spearmanr(
    a[mask],
    b[mask]
)

print(f"N paired samples = {mask.sum()}")
print(f"Pearson r        = {pearson_r:.6f}")
print(f"Pearson P        = {pearson_p:.6g}")
print(f"Spearman rho     = {spearman_rho:.6f}")
print(f"Spearman P       = {spearman_p:.6g}")


print("\n" + "=" * 70)
print("ABCA6 MEAN EXPRESSION")
print("=" * 70)

print(
    "ILMN_1701551:",
    np.nanmean(found["ABCA6_probe1"])
)

print(
    "ILMN_1795507:",
    np.nanmean(found["ABCA6_probe2"])
)

print("\nCOMPLETE")
