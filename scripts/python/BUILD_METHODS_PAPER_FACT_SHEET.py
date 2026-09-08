#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

ROOT = Path.home() / "project_ml"

OUT = (
    ROOT
    / "06_Manuscript"
    / "METHODS_PAPER_CANONICAL_FACT_SHEET.txt"
)

text = r"""
================================================================================
METHODS PAPER — CANONICAL FACT SHEET
================================================================================

Working title
-------------
Discrimination without validation: batch confounding and calibration failure
in transcriptomic signature transfer


CENTRAL STUDY QUESTION
----------------------
Can apparent external discrimination of a transcriptomic disease signature
be interpreted as independent biological validation when the external cohort
contains dataset-specific confounding and calibration failure?


STUDY DESIGN
------------
Discovery dataset:
    GSE48350

External transfer-test dataset:
    GSE5281

Important terminology:
    "cross-dataset transfer test"

Do NOT use:
    "independent external validation"

unless referring explicitly to the historical analysis terminology.


FINAL THREE-GENE SIGNATURE
--------------------------
ABCA6
CRLF1
TNFRSF11B


MODEL DEVELOPMENT
-----------------
Training ROC-AUC:
    0.854902

Held-out test ROC-AUC:
    0.694444

Repeated nested cross-validation:
    approximately 0.757 ± 0.165


GSE5281 HIPPOCAMPAL TRANSFER TEST
---------------------------------
Total hippocampal samples:
    23

AD:
    10

Control:
    13

Raw external ROC-AUC:
    0.661538

Harmonized external ROC-AUC:
    0.684615

External bootstrap 95% interval:
    approximately 0.417–0.877

External permutation:
    raw and harmonized analyses performed

Important:
    External AUC must NOT be presented as clean independent biological
    validation because diagnosis is perfectly confounded with submission
    cohort in the hippocampal subset.


GSE5281 SUBMISSION STRUCTURE
----------------------------
Entire GSE5281 dataset:
    161 samples

Submission cohorts:
    74 samples — Jul 10 2006
    87 samples — Oct 19 2007

Hippocampal controls:
    GSM119628–GSM119640
    Jul 10 2006

Hippocampal AD:
    GSM238799–GSM238808
    Oct 19 2007


DESIGN IDENTIFIABILITY
----------------------
Design:
    Intercept + Diagnosis + Submission cohort

Design dimensions:
    23 × 3

Matrix rank:
    2

Number of columns:
    3

Rank deficient:
    TRUE

Diagnosis equals submission indicator:
    TRUE

Formal interpretation:
    Diagnosis and submission cohort are perfectly confounded.

    Independent estimation of diagnosis and submission-cohort effects
    is therefore impossible within this 23-sample subset.


CLAIM BOUNDARIES
----------------
SUPPORTED:

1. Diagnosis is perfectly confounded with submission cohort.

2. The diagnosis + submission design is rank deficient.

3. Independent estimation of diagnosis and submission-cohort effects is
   impossible within the hippocampal subset.

4. The GSE5281 analysis should be described as a cross-dataset transfer test.

5. External discrimination should not be interpreted as clean independent
   biological validation.

6. The model exhibits external calibration failure.

7. Biological plausibility of individual genes can be evaluated separately
   from validation performance.


NOT SUPPORTED:

1. The entire external AUC is caused by technical batch.

2. The three-gene signature has no biological signal.

3. The three-gene model is superior to ABCA6 + TNFRSF11B externally.

4. The signature is clinically validated.

5. All three genes are individually necessary in every cohort.

6. The signature clearly improves discrimination over cell-type composition.

7. TNFRSF11B is directionally conserved across brain and blood.


EXTERNAL MODEL COMPARISON
-------------------------
ABCA6:
    external AUC approximately 0.662

ABCA6 + TNFRSF11B:
    external AUC approximately 0.731

ABCA6 + CRLF1 + TNFRSF11B:
    external AUC approximately 0.662

Therefore:
    The full three-gene model is NOT demonstrably superior to the
    strongest two-gene alternative.


CALIBRATION
-----------
External discrimination and calibration diverged.

The locked model assigned near-zero AD probabilities to the external
samples and classified every external sample as control at the locked
threshold.

Therefore:
    Rank discrimination should not be conflated with useful probability
    calibration or threshold-based classification.


BIOLOGICAL INTERPRETATION
-------------------------
ABCA6:
    Directionally supported across evaluated datasets.
    Associated with oligodendrocyte composition.
    Independent single-cell evidence supports biological relevance.

CRLF1:
    Directionally conserved across evaluated datasets.

TNFRSF11B:
    Directionally positive in training and hippocampal data but negative
    in blood.

Therefore:
    Cross-tissue biological transportability is partial rather than complete.


CELL COMPOSITION
----------------
Signature independence from bulk cell-type composition:
    Supported by existing analysis.

Oligodendrocyte composition:
    Not sufficient to explain the entire signature signal.

However:
    Clear superiority of the signature over composition was NOT supported.


AGE
---
Existing analysis supports information beyond age in the development data.

However:
    Residual age structure in external discrimination cannot be excluded.


STABILITY
---------
Nested feature-selection stability did NOT establish all three genes as
highly stable individually.

Reported approximate nested selection frequencies:
    TNFRSF11B: 0.628
    CRLF1:     0.504
    ABCA6:     0.440

Interpretation:
    Stability evidence should be attenuated.


PRIMARY METHODS-PAPER MESSAGE
-----------------------------
External performance is not equivalent to external validation.

A transfer test can show apparent discrimination while failing to provide
an independently identifiable estimate of disease-associated signal when
diagnosis is perfectly confounded with dataset-specific cohort structure.

Calibration failure provides a second independent warning:
rank-based discrimination may persist while probability estimates become
non-transportable.


RECOMMENDED FRAMING
-------------------
This is a methodological cautionary case study using an Alzheimer's disease
transcriptomic signature as a worked example.

The paper should emphasize:

    1. external validation design,
    2. identifiability,
    3. batch/cohort confounding,
    4. calibration,
    5. discrimination,
    6. cell-composition effects,
    7. biological plausibility,
    8. appropriate claim boundaries.


DO NOT OVERCLAIM
----------------
Do not state that GSE5281 is "entirely technical."

Do not state that the signature is "invalid."

Do not state that the genes have no biological relevance.

The scientifically defensible conclusion is that the GSE5281 hippocampal
subset cannot independently distinguish diagnosis from submission-cohort
effects and therefore cannot serve as clean independent biological
validation of the signature.


================================================================================
END CANONICAL FACT SHEET
================================================================================
"""

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w") as f:
    f.write(
        "Generated: "
        + datetime.now().isoformat()
        + "\n"
    )
    f.write(text)

print("=" * 80)
print("CANONICAL METHODS-PAPER FACT SHEET CREATED")
print("=" * 80)
print(OUT)
