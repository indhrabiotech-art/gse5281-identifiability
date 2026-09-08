from pathlib import Path
import re

ROOT = Path("06_Manuscript")

files = {
    "Abstract": ROOT / "Abstract_3gene_signature.txt",
    "Methods": ROOT / "Methods" / "Methods_3gene_signature.txt",
    "Results": ROOT / "Results" / "Results_3gene_signature.txt",
    "Discussion": ROOT / "Discussion" / "Discussion_3gene_signature.txt",
    "Figure legends": ROOT / "Results" / "Figure_Legends_3gene_signature.txt"
}

print("=" * 80)
print("FINAL MANUSCRIPT SCIENTIFIC QC")
print("=" * 80)

texts = {}

for name, path in files.items():
    if path.exists():
        texts[name] = path.read_text()
        print(f"FOUND    {name:20s} {path}")
    else:
        print(f"MISSING  {name:20s} {path}")

combined = "\n".join(texts.values())

print("\n" + "=" * 80)
print("CORE NUMERICAL CLAIMS")
print("=" * 80)

checks = {
    "ABCA6": "ABCA6" in combined,
    "CRLF1": "CRLF1" in combined,
    "TNFRSF11B": "TNFRSF11B" in combined,

    "Training AUC 0.855":
        bool(re.search(r"0\.855|0\.8549", combined)),

    "Internal test AUC 0.694":
        bool(re.search(r"0\.694", combined)),

    "External AUC 0.685":
        bool(re.search(r"0\.685|0\.6846", combined)),

    "Bootstrap CI":
        bool(re.search(r"0\.442.*0\.886|0\.4417.*0\.8864", combined)),

    "Permutation p":
        bool(re.search(r"0\.1447|0\.144686", combined)),

    "External n=23":
        "23" in combined
}

for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL':8s} {name}")

print("\n" + "=" * 80)
print("OVERCLAIMING CHECK")
print("=" * 80)

dangerous_phrases = [
    "clinically validated",
    "clinical biomarker",
    "diagnostic biomarker",
    "diagnostic test",
    "causes Alzheimer's",
    "causes Alzheimer",
    "proves that",
    "proven mechanism",
    "therapeutic target"
]

found = False

for phrase in dangerous_phrases:
    if phrase.lower() in combined.lower():
        print(f"REVIEW   {phrase}")
        found = True

if not found:
    print("PASS     No obvious high-risk overclaiming phrases detected.")

print("\n" + "=" * 80)
print("HARMONIZATION LANGUAGE CHECK")
print("=" * 80)

harm_checks = {
    "empirical quantile mapping":
        "empirical quantile mapping" in combined.lower(),

    "GSE48350 reference":
        "GSE48350" in combined,

    "external labels not used":
        "diagnostic labels" in combined.lower(),

    "raw vs harmonized":
        "raw" in combined.lower() and "harmonized" in combined.lower(),

    "AUC robustness":
        "0.662" in combined and "0.685" in combined
}

for name, ok in harm_checks.items():
    print(f"{'PASS' if ok else 'FAIL':8s} {name}")

print("\n" + "=" * 80)
print("MANUSCRIPT FILE SIZES")
print("=" * 80)

for name, path in files.items():
    if path.exists():
        chars = len(path.read_text())
        words = len(path.read_text().split())
        print(f"{name:20s} {words:5d} words | {chars:7d} characters")

print("\n" + "=" * 80)
print("FINAL MANUSCRIPT QC COMPLETE")
print("=" * 80)
