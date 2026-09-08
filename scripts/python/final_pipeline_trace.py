from pathlib import Path

print("=" * 85)
print("FINAL 3-GENE PIPELINE TRACE")
print("=" * 85)

stages = {
    "1. Preprocessing":
        [
            "R_scripts/preprocess_raw_CEL.R",
            "R_scripts/annotate_RMA_genelevel.R",
            "R_scripts/qc_filter_split.R"
        ],

    "2. Training preprocessing":
        [
            "R_scripts/training_top25_variance_filter.R",
            "R_scripts/age_adjust_training_only.R"
        ],

    "3. Feature selection / model development":
        [
            "Python_scripts/lasso_feature_selection.py",
            "Python_scripts/final_3gene_lasso_validation.py"
        ],

    "4. Internal validation":
        [
            "Python_scripts/internal_test_validation.py",
            "Python_scripts/final_3gene_characterization.py"
        ],

    "5. External validation":
        [
            "Python_scripts/validate_3gene_external.py",
            "Python_scripts/validate_3gene_external_harmonized.py"
        ],

    "6. External robustness":
        [
            "Python_scripts/bootstrap_external_auc.py",
            "Python_scripts/permutation_external_auc.py",
            "Python_scripts/audit_harmonization_method.py",
            "Python_scripts/robustness_raw_vs_harmonized_3gene.py"
        ],

    "7. Biological characterization":
        [
            "Python_scripts/prepare_3gene_biology_input.py",
            "Python_scripts/enrichment_3gene.py",
            "Python_scripts/string_first_shell_3gene.py",
            "Python_scripts/build_final_biology_summary.py"
        ],

    "8. Final figures":
        [
            "Python_scripts/generate_final_figures.py",
            "Python_scripts/generate_3gene_expression_heatmap.py",
            "Python_scripts/generate_figure_legends.py"
        ],

    "9. Manuscript":
        [
            "Python_scripts/generate_manuscript_abstract.py",
            "Python_scripts/generate_manuscript_methods.py",
            "Python_scripts/generate_manuscript_results.py",
            "Python_scripts/generate_manuscript_discussion.py"
        ]
    }

failed = []

for stage, scripts in stages.items():

    print("\n" + "=" * 85)
    print(stage)
    print("-" * 85)

    for script in scripts:

        exists = Path(script).exists()

        print(
            f"{'FOUND' if exists else 'MISSING':8s} {script}"
        )

        if not exists:
            failed.append(script)

print("\n" + "=" * 85)
print("FINAL SIGNATURE")
print("=" * 85)

print("ABCA6")
print("CRLF1")
print("TNFRSF11B")

print("\n" + "=" * 85)
print("FINAL VALIDATION")
print("=" * 85)

print("Training AUC       : 0.854902")
print("Internal test AUC  : 0.694444")
print("External AUC       : 0.684615")
print("Bootstrap 95% CI   : 0.441667 – 0.886364")
print("Permutation p      : 0.144686")
print("External n         : 23")

print("\n" + "=" * 85)

if failed:
    print("PIPELINE TRACE STATUS: REVIEW REQUIRED")
    print("\nMissing scripts:")
    for f in failed:
        print(" -", f)
else:
    print("PIPELINE TRACE STATUS: PASS")
    print("All designated final-pipeline scripts are present.")

print("=" * 85)
