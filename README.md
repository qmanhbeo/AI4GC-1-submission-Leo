# Replication: Global Sustainability Performance and Regional Disparities

**Paper:** Celik et al. (2025), *MDPI Sustainability 17, 7411*  
**DOI:** https://doi.org/10.3390/su17167411  
**Original repository:** https://github.com/Sadullah4535/Global-Sustainability-Performance-  
**Replication by:** Leonardo Manh Nguyen  
**Programme:** MSc AI & Sustainable Development, University of Birmingham

This repository is the cleaned submission package for the assignment. It contains the original author code, a refactored author-aligned version, a notebook version, a batch-safe replication script, the source dataset, the paper PDF, the written report, and the generated replication outputs.

Works on Windows and Linux. Haven't tested for MacOS.

## Quickstart for Marking

If you only want to assess the submission without rerunning code:

1. Read `report.md` for the replication and critical assessment.
2. Open `replication_results/` for the generated figures, tables, and `run_summary.txt`.
3. Open `README.md` for the repository structure and file roles.

If you want to rerun the full replication from the command line, the easiest route is the cleaned script.

```bash
git clone https://github.com/qmanhbeo/AI4GC-1-submission-Leo
cd AI4GC-1-submission-Leo
```

Use either `venv` or Conda:

```bash
# Option A: venv
python -m venv .venv
# activate the virtual environment in your shell
python -m pip install -r requirements.txt
python 2_replication.py
```

```bash
# Option B: conda
conda create -n gsp-replication python=3.11 -y
conda activate gsp-replication
python -m pip install -r requirements.txt
python 2_replication.py
```

This regenerates the figures, tables, CSV exports, and `replication_results/run_summary.txt`.

If you prefer a notebook view instead of the batch script:

```bash
jupyter notebook 1_author_original_refactored_notebook.ipynb
```

Then run all cells from top to bottom.

## Current Project Structure

```text
AI4GC-1-submission/
|-- 1_author_original_refactored.py
|-- 1_author_original_refactored_notebook.ipynb
|-- 2_replication.py
|-- README.md
|-- report.md
|-- requirements.txt
|-- SDG2025.csv
|-- paper.pdf
|-- original-repo/
|   |-- Codes.py
|   |-- README.md
|   `-- SDG2025.csv
`-- replication_results/
    |-- fig1_circular_flow.png
    |-- fig2_top_bottom20_sdg.png
    |-- fig3_top20_indicators.png
    |-- fig4_correlation.png
    |-- fig4_correlation_values.csv
    |-- fig5_elbow_silhouette.png
    |-- fig6_3d_pca_clusters.png
    |-- fig7_cluster_heatmap.png
    |-- fig7_cluster_means_normalized.csv
    |-- fig8_feature_importance.png
    |-- fig8_feature_importance_values.csv
    |-- fig9_confusion_matrices.png
    |-- fig10_roc_curves.png
    |-- table2_elbow.csv
    |-- table3_anova.csv
    |-- table3_anova_computed.csv
    |-- table3_anova_verify.csv
    |-- table4_manova.csv
    |-- table4_manova_computed.csv
    |-- table4_manova_verify.csv
    |-- table5_classification.csv
    |-- table5_classification_computed.csv
    `-- run_summary.txt
```

## Code Structure and File Roles

The main code files form a progression from the original author release to the cleaned replication used for the assignment.

| File name | What stays the same | What changes that affect results or reproducibility |
| --- | --- | --- |
| `original-repo/Codes.py` | This is the original source workflow: the same CSV, the same five numeric variables, `StandardScaler`, PCA, K-Means around `k=5`, and later classifier evaluation blocks are already present here. | It is a notebook dump with repeated and competing code paths, state-dependent execution, no explicit `n_init` in K-Means, full-data pairwise correlations for Figure 4, and manually typed ANOVA and MANOVA tables. It is the least auditable version. |
| `1_author_original_refactored.py` | It preserves the author's effective sample construction and overall pipeline while putting the figures and tables into one ordered script. | It makes the clustering setup explicit with `n_init=1`, uses complete-case correlations for Figure 4 via `df_clean[features].corr()`, adds computed ANOVA and MANOVA verification exports, and fixes the runnability problems in the raw notebook dump. |
| `1_author_original_refactored_notebook.ipynb` | It is the notebook form of `1_author_original_refactored.py` and follows the same code path. | There are no intended result-affecting differences from `1_author_original_refactored.py`; the difference is notebook presentation, outputs, and metadata. |
| `2_replication.py` | It keeps the same dataset, five-feature clustering setup, `StandardScaler`, PCA(3), `k=5`, and the same family of downstream classifiers. | It is the cleanest version for marking and rerunning: it is batch-safe, keeps explicit paper-style K-Means settings, switches Figure 4 back to full-data pairwise correlations `df[FEATURES].corr()`, separates paper-reported tables from computed verification tables, treats `cluster` as categorical in MANOVA with `C(cluster)`, and writes extra CSV outputs plus a run summary. |

## Recommended Files to Inspect

| File or folder | Why it matters |
| --- | --- |
| `report.md` | Main write-up for the assignment: replication findings and critical assessment. |
| `2_replication.py` | Simplest one-command script to rerun the full workflow. |
| `1_author_original_refactored_notebook.ipynb` | Same refactored workflow in notebook form. |
| `replication_results/` | Generated outputs used for marking: figures, tables, verification CSVs, and run summary. |
| `original-repo/` | Archived copy of the author's released materials for comparison. |

## Notes on Reproducibility

- The CSV contains 167 countries, but the clustering workflow drops rows with missing values in the five modelling features, leaving 143 countries in the effective clustering sample.
- The paper-aligned clustering setup depends on `KMeans(..., random_state=42, n_init=1)`. This is made explicit in the cleaned scripts.
- `2_replication.py` writes both paper-reported tables and computed verification tables so the differences are easy to audit.
- `replication_results/run_summary.txt` is the fastest single file to open if you want a concise summary of what the script produced.
