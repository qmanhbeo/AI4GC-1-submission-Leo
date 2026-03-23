# Replication: Global Sustainability Performance and Regional Disparities

**Paper:** Çelik et al. (2025), *MDPI Sustainability 17, 7411*  
**DOI:** https://doi.org/10.3390/su17167411  
**Author's GitHub repo**: https://github.com/Sadullah4535/Global-Sustainability-Performance-  
**Replication by:** Leonardo Manh Nguyen (MSc AI & Sustainable Development, University of Birmingham)  
**Date:** 14 March 2026

---

## Files

| File | Description |
|------|-------------|
| `Codes_replicated_cleaned.ipynb` | Main replication notebook |
| `Codes.py` | Original code from the authors' GitHub repo |
| `SDG2025.csv` | Dataset from the authors' GitHub repo |
| `requirements.txt` | Python dependencies |
| `paper.pdf` | Original paper |
| `replication_results/` | All output figures (fig1–fig10) and tables (table2–table5) as PNG/CSV |

---

## How to Run

**1. Create and activate the conda environment:**
```bash
conda create -n gsp python=3.11 -y
conda activate gsp
pip install -r requirements.txt
```

**2. Open and run the notebook:**

Open `Codes_replicated_cleaned.ipynb` in either Jupyter or VS Code 

```bash
# If run in Jupyter Notebook:
jupyter notebook Codes_replicated_cleaned.ipynb
```
and run all cells top to bottom (`Kernel → Restart & Run All` in Jupyter, or `Run All` in VS Code).

All figures and tables will be saved to `replication_results/`.

---

## Replication Notes

- The authors' repository CSV contains **167 countries**, but 24 have missing values across the 5 features. After `dropna()`, the working sample is **143 countries** vs. **166** in the paper.
- This causes minor numerical differences in correlations, WCSS values, and silhouette scores. All **qualitative findings are preserved**: k=5 remains optimal, cluster structure and feature importance rankings match the paper.
- If we impute NA by median, replicated Figure 4 is similar to the paper's. Simplying dropping NAs gives different numbers.  
- Tables 3 & 4 (ANOVA/MANOVA) were **hardcoded in the original authors' code**, not computed. This replication computes them properly from data using `scipy` and `statsmodels` — see the `_verify.csv` exports.
