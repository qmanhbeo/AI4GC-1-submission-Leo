import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.multivariate.manova import MANOVA
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

OUTDIR = Path("replication_results")
OUTDIR.mkdir(exist_ok=True)

FEATURES = ["sdg_score", "spillover_score", "regional_score", "population", "progress"]
PAPER_KMEANS = {"init": "k-means++", "n_init": 1, "random_state": 42}

# Load data and prepare clustering inputs
df = pd.read_csv("SDG2025.csv", sep=";", encoding="ISO-8859-1", engine="python")
df = df.rename(
    columns={
        "Country": "country",
        "2025 SDG Index Score": "sdg_score",
        "International Spillovers Score (0-100)": "spillover_score",
        "Regional Score (0-100)": "regional_score",
        "Population in 2024": "population",
        "Progress on Headline SDGi (p.p.)": "progress",
        "Regions used for the SDR": "region",
    }
)

x = df[FEATURES].dropna()
df_clean = df.loc[x.index].copy()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaled)

labels = KMeans(n_clusters=5, **PAPER_KMEANS).fit_predict(x_scaled)
df_clean["cluster"] = labels
df_clean["PCA1"] = x_pca[:, 0]
df_clean["PCA2"] = x_pca[:, 1]
df_clean["PCA3"] = x_pca[:, 2]

# Figure 1
steps = [
    "1. Data\nPreprocessing",
    "2. Exploratory\nAnalysis",
    "3. Dimensionality\nReduction",
    "4. Clustering",
    "5. Cluster\nValidation",
    "6. Interpretability",
    "7. Model\nTesting",
    "8. Performance\nEvaluation",
]
angles = np.linspace(0, 2 * np.pi, len(steps), endpoint=False)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines["polar"].set_visible(False)
ax.set_facecolor("white")

for angle, step in zip(angles, steps):
    ax.text(
        angle,
        1.0,
        step,
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#cce5ff",
            "edgecolor": "steelblue",
            "linewidth": 1.2,
        },
    )

for i in range(len(steps)):
    a1, a2 = angles[i], angles[(i + 1) % len(steps)]
    ax.annotate(
        "",
        xy=(a2, 0.78),
        xytext=(a1, 0.78),
        arrowprops={
            "arrowstyle": "->",
            "color": "steelblue",
            "lw": 1.8,
            "connectionstyle": "arc3,rad=0.15",
        },
    )

plt.title("Circular Methodological Flow", fontsize=13, weight="bold", pad=25)
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_circular_flow.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2
fig_df = df[["country", "sdg_score"]].dropna().sort_values("sdg_score", ascending=False)
top20 = fig_df.head(20)
bottom20 = fig_df.tail(20).sort_values("sdg_score")

fig, axes = plt.subplots(1, 2, figsize=(14, 9))
sns.barplot(data=top20, y="country", x="sdg_score", palette="Greens_r", ax=axes[0])
axes[0].set_title("Top 20 Countries by 2025 SDG Score", fontsize=12, weight="bold")
axes[0].set_xlabel("SDG Score")
axes[0].set_ylabel("")

sns.barplot(data=bottom20, y="country", x="sdg_score", palette="Reds_r", ax=axes[1])
axes[1].set_title("Bottom 20 Countries by 2025 SDG Score", fontsize=12, weight="bold")
axes[1].set_xlabel("SDG Score")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(OUTDIR / "fig2_top_bottom20_sdg.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3
variables = ["spillover_score", "regional_score", "population", "progress"]
titles = [
    "Top 20 Countries by International Spillover Score",
    "Top 20 Countries by Regional Score",
    "Top 20 Countries by Population (2024)",
    "Top 20 Countries by SDG Progress",
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for ax, var, title in zip(axes, variables, titles):
    top20 = df[["country", var]].dropna().sort_values(var, ascending=False).head(20)
    sns.barplot(data=top20, x="country", y=var, palette="viridis", ax=ax)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

plt.suptitle(
    "Figure 3. Top 20 countries by key Sustainable Development Indicators.",
    fontsize=13,
    weight="bold",
)
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_top20_indicators.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4
corr = df[FEATURES].corr()
corr.reset_index().to_csv(OUTDIR / "fig4_correlation_values.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "fig4_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# Table 2 and Figure 5
rows = []
ks = list(range(2, 11))
for k in ks:
    km = KMeans(n_clusters=k, **PAPER_KMEANS)
    k_labels = km.fit_predict(x_scaled)
    rows.append(
        {
            "k": k,
            "WCSS (Inertia)": km.inertia_,
            "Silhouette Score": silhouette_score(x_scaled, k_labels),
        }
    )

table2 = pd.DataFrame(rows)
table2.to_csv(OUTDIR / "table2_elbow.csv", index=False)

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(table2["k"], table2["WCSS (Inertia)"], "o-", color="steelblue", label="WCSS (Inertia)")
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("WCSS (Inertia)", color="steelblue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.plot(table2["k"], table2["Silhouette Score"], "o-", color="crimson", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="crimson", fontsize=11)
ax2.tick_params(axis="y", labelcolor="crimson")

ax1.axvline(x=5, color="green", linestyle="--", linewidth=1.5, label="k=5 (optimal)")
ax1.set_title("Elbow Method and Silhouette Score", fontsize=12, weight="bold")
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(OUTDIR / "fig5_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 6
cluster_colors = ["#FFD700", "#FF4500", "#32CD32", "#1E90FF", "#800080"]
cluster_counts = df_clean["cluster"].value_counts().sort_index()

kmeans5 = KMeans(n_clusters=5, **PAPER_KMEANS)
labels5 = kmeans5.fit_predict(x_scaled)
centers_pca = pca.transform(kmeans5.cluster_centers_)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df_clean["PCA1"],
    df_clean["PCA2"],
    df_clean["PCA3"],
    c=[cluster_colors[l] for l in labels5],
    s=45,
    alpha=0.85,
)

for _, row in df_clean.iterrows():
    cid = int(row["cluster"])
    txt = ax.text(
        row["PCA1"],
        row["PCA2"],
        row["PCA3"],
        row["country"],
        fontsize=7,
        color=cluster_colors[cid],
    )
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])

ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    centers_pca[:, 2],
    c="black",
    marker="X",
    s=220,
    label="Centroids",
)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title(
    "3D Visualization of KMeans Clusters (k=5) with Country Labels\n"
    f"Silhouette={silhouette_score(x_scaled, labels5):.4f}",
    fontsize=12,
    weight="bold",
)

handles = [
    plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor=cluster_colors[i],
        markersize=10,
        label=f"Cluster {i} (n={cluster_counts[i]})",
    )
    for i in sorted(cluster_counts.index)
]
handles.append(mpatches.Patch(color="black", label="Centroids"))
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

plt.tight_layout()
plt.savefig(OUTDIR / "fig6_3d_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 7
cluster_means = df_clean.groupby("cluster")[FEATURES].mean()
mm = MinMaxScaler()
cluster_means_norm = pd.DataFrame(
    mm.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index,
)
cluster_means_norm.reset_index().to_csv(OUTDIR / "fig7_cluster_means_normalized.csv", index=False)

plt.figure(figsize=(9, 5))
sns.heatmap(cluster_means_norm, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5)
plt.title("Normalized Average Feature Scores per Cluster", fontsize=13, weight="bold")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUTDIR / "fig7_cluster_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Paper Tables 3 and 4
paper_table3 = pd.DataFrame(
    {
        "Feature": ["PCA1", "PCA2"],
        "F-Value": [45.632, 39.871],
        "p-Value": ["1.23e-10", "3.45e-09"],
        "Significant (p<0.05)": ["Yes", "Yes"],
    }
)

paper_table4 = pd.DataFrame(
    {
        "Test Statistic": [
            "Wilks' Lambda",
            "Pillai's Trace",
            "Hotelling-Lawley Trace",
            "Roy's Largest Root",
        ],
        "Value": [0.243, 0.573, 1.047, 0.823],
        "F-Value": [67.32, 70.89, 72.10, 69.55],
        "p-Value": [0.000, 0.000, 0.000, 0.000],
        "Significant (p<0.05)": ["Yes", "Yes", "Yes", "Yes"],
    }
)

paper_table3.to_csv(OUTDIR / "table3_anova.csv", index=False)
paper_table4.to_csv(OUTDIR / "table4_manova.csv", index=False)

# Computed Tables 3 and 4
pca_df = pd.DataFrame(x_pca[:, :2], columns=["PCA1", "PCA2"])
pca_df["cluster"] = labels

anova_rows = []
for comp in ["PCA1", "PCA2"]:
    groups = [grp[comp].values for _, grp in pca_df.groupby("cluster")]
    f_val, p_val = f_oneway(*groups)
    anova_rows.append(
        {
            "Feature": comp,
            "F-Value": round(f_val, 3),
            "p-Value": f"{p_val:.2e}",
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
        }
    )

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUTDIR / "table3_anova_computed.csv", index=False)

mv = MANOVA.from_formula("PCA1 + PCA2 ~ C(cluster)", data=pca_df)
effect = mv.mv_test().results["C(cluster)"]["stat"]
stat_map = {
    "Wilks' lambda": "Wilks' Lambda",
    "Pillai's trace": "Pillai's Trace",
    "Hotelling-Lawley trace": "Hotelling-Lawley Trace",
    "Roy's greatest root": "Roy's Largest Root",
}

manova_rows = []
for raw_name, display_name in stat_map.items():
    row = effect.loc[raw_name]
    manova_rows.append(
        {
            "Test Statistic": display_name,
            "Value": round(float(row["Value"]), 3),
            "F-Value": round(float(row["F Value"]), 3),
            "p-Value": round(float(row["Pr > F"]), 3),
            "Significant (p<0.05)": "Yes" if float(row["Pr > F"]) < 0.05 else "No",
        }
    )

manova_df = pd.DataFrame(manova_rows)
manova_df.to_csv(OUTDIR / "table4_manova_computed.csv", index=False)

# Figure 8
x_rf = df_clean[FEATURES]
y_rf = df_clean["cluster"]

x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(
    x_rf, y_rf, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_rf, y_train_rf)

train_acc = accuracy_score(y_train_rf, rf.predict(x_train_rf))
test_acc = accuracy_score(y_test_rf, rf.predict(x_test_rf))

feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values(
    "Importance", ascending=True
)
feat_imp.sort_values("Importance", ascending=False).to_csv(
    OUTDIR / "fig8_feature_importance_values.csv", index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    feat_imp["Feature"],
    feat_imp["Importance"],
    color=["#2ecc71" if i < 2 else "#3498db" for i in range(len(feat_imp))],
    edgecolor="white",
)

for bar, val in zip(bars, feat_imp["Importance"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)

ax.set_xlabel("Importance", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_xlim(0, feat_imp["Importance"].max() + 0.07)
ax.set_title(
    f"Training Accuracy: {train_acc:.3f}  |  Test Accuracy: {test_acc:.3f}",
    fontsize=11,
    bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUTDIR / "fig8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Paper Table 5
paper_table5 = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "SVM",
            "Decision Tree",
            "XGBoost",
            "ANN",
            "Logistic Regression",
        ],
        "Accuracy": [0.977, 0.977, 0.953, 0.907, 0.977, 0.953],
        "Precision (Macro Avg)": [0.789, 0.982, 0.948, 0.750, 0.982, 0.968],
        "Recall (Macro Avg)": [0.800, 0.960, 0.976, 0.708, 0.988, 0.968],
        "F1 Score (Macro Avg)": [0.794, 0.968, 0.960, 0.718, 0.984, 0.968],
        "ROC AUC (Macro)": [1.000, 1.000, 0.974, 0.835, 1.000, 0.994],
    }
)
paper_table5.to_csv(OUTDIR / "table5_classification.csv", index=False)

# Computed Table 5
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, labels, test_size=0.3, random_state=42, stratify=labels
)

classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    ),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    clf = OneVsRestClassifier(model)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_test)
    else:
        try:
            scores = clf.decision_function(x_test)
            if scores.ndim == 1:
                y_prob = np.vstack([1 - scores, scores]).T
            else:
                exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                y_prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        except Exception:
            y_prob = np.zeros((x_test.shape[0], n_classes))

    if y_prob.shape[1] < n_classes:
        prob_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for c in classes:
            if c not in prob_df.columns:
                prob_df[c] = 0.0
        y_prob = prob_df[classes].values

    y_prob = np.nan_to_num(y_prob)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = np.nan

    results[name] = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred),
    }

rows = []
for name, result in results.items():
    rep = result["report"]
    rows.append(
        {
            "Model": name,
            "Accuracy": round(rep.get("accuracy", np.nan), 3),
            "Precision (Macro Avg)": round(rep.get("macro avg", {}).get("precision", np.nan), 3),
            "Recall (Macro Avg)": round(rep.get("macro avg", {}).get("recall", np.nan), 3),
            "F1 Score (Macro Avg)": round(rep.get("macro avg", {}).get("f1-score", np.nan), 3),
            "ROC AUC (Macro)": round(result["roc_auc"], 3),
        }
    )

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUTDIR / "table5_classification_computed.csv", index=False)

# Figure 9
palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds", "coolwarm"]
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    sns.heatmap(
        result["conf_matrix"],
        annot=True,
        fmt="d",
        cmap=palettes[idx],
        ax=axes[idx],
        cbar=False,
        linewidths=0.8,
        linecolor="black",
    )
    axes[idx].set_title(f"{name} - Confusion Matrix", fontsize=11, weight="bold")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

plt.suptitle("Figure 9. Confusion Matrices", fontsize=13, weight="bold", y=1.002)
plt.tight_layout()
plt.savefig(OUTDIR / "fig9_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 10
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    y_prob = result["y_prob"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        try:
            auc_val = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        except Exception:
            auc_val = np.nan
        ax.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} - Macro AUC: {result['roc_auc']:.3f}", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.suptitle("Figure 10. ROC Curves", fontsize=13, weight="bold", y=1.002)
plt.tight_layout()
plt.savefig(OUTDIR / "fig10_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Run summary
summary_lines = [
    "Paper-aligned replication completed.",
    f"Output folder: {OUTDIR}",
    f"Rows in CSV: {df['country'].nunique()}",
    f"Rows used for clustering: {len(x)}",
    f"Missing progress values dropped: {df['progress'].isna().sum()}",
    f"KMeans settings: {PAPER_KMEANS}",
    "",
    "Cluster sizes (k=5):",
    str(df_clean["cluster"].value_counts().sort_index().to_dict()),
    "",
    "Table 2 / Figure 5 (paper-aligned):",
    table2.round(6).to_string(index=False),
    "",
    "Figure 4 correlation matrix (full dataframe):",
    corr.round(3).to_string(),
    "",
    "Computed Table 5 verification:",
    metrics_df.to_string(index=False),
]
(OUTDIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
