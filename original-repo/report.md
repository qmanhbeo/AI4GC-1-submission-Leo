## A) Report: Replication and Critical Assessment

### Introduction

This report replicates and critically appraises Celik et al. (2025), a study that uses machine learning on the 2025 SDG Index to group countries by sustainability performance. The topic is relevant because sustainability indicators are complex and high-dimensional, so clustering can potentially reveal useful policy patterns. The paper combines K-Means, PCA, and supervised classification, which makes it a good case for testing reproducibility.

My main finding is that the paper's central five-cluster result is only partly reproducible from the released repository. The broad clustering pattern can be recovered, but only after reconstructing missing implementation details from a messy notebook-style script. The repository does not provide a clean one-command pipeline, key settings for Figure 5 are under-specified, some later results are hard-coded rather than computed, and the Random Forest results still do not fully match the paper. The study asks an important question, but its release standard is weak.

### Summary of the Original Paper

The paper studies country-level sustainability differences using the 2025 SDG Index. It uses cross-sectional data and focuses on SDG score, international spillover score, regional score, population, and progress on headline SDG indicators. In the written methods, the paper also refers to region as part of the modelling process. The aim is to identify groups of countries with similar sustainability profiles and then show that these groups are statistically meaningful.

The main method is K-Means clustering. The number of clusters is chosen using the Elbow method and silhouette score, with the paper selecting five clusters. PCA is then used to visualise the clusters, and average feature profiles are used to interpret them. The paper also reports ANOVA and MANOVA results as evidence that the clusters differ significantly.

To strengthen the argument, the authors train several classifiers, including Random Forest, SVM, Decision Tree, XGBoost, ANN, and Logistic Regression, to predict the cluster labels. They report high predictive performance and use ROC curves and feature importance to argue that the groupings are stable and policy-relevant.

### Summary of My Replication

I worked only from the materials in the released repository: a minimal `README.md`, one CSV file, and one large exported notebook script (`Codes.py`). The repository is not a clean replication package. The README gives no instructions or package versions, and the main script is a notebook dump with repeated cells and state-dependent execution. It does not run cleanly as a standalone script because it contains raw text that causes a syntax error and because some sections rely on variables created elsewhere in the notebook.

I therefore rebuilt the workflow step by step. First, I checked the effective sample. The CSV contains 167 countries, but the clustering code drops rows with missing values in the five numerical features, leaving 143 observations. This is important because the paper refers to a larger country count (166), while the published cluster sizes actually sum to 143.

The main technical focus of my replication was Figure 5 and the choice of five clusters. In the released code, K-Means is run with `random_state=42`, but important initialization details, especially `n_init`, are not specified. That omission matters because WCSS and silhouette values are sensitive to K-Means initialization and to scikit-learn version. After aligning the preprocessing in the code, I found that the paper's cluster-selection pattern is broadly recoverable, but only once this missing implementation detail is restored.

I then reviewed the validation sections. The visual cluster profiling can be reconstructed from the code, but the ANOVA and MANOVA tables cannot be independently replicated because the script prints manually hard-coded values instead of calculating them. The classification section is also hard to audit because the repository contains several overlapping model-evaluation blocks with slightly different setups. Most of the classification story is similar, but the Random Forest results still do not fully match the paper.

### Critical Assessment of the Research Design

The paper uses a cross-sectional, observational design based on secondary country-level data. That is reasonable for descriptive pattern detection, and K-Means plus PCA is a plausible toolkit for summarising sustainability indicators. The problem is not the general idea of the design, but the weak transparency of its implementation.

The first weakness is repository-level reproducibility. The released materials do not meet the standard of a clean computational study. There is no meaningful README, no dependency file, no pinned software environment, and no clear instruction for reproducing tables or figures. This matters because Figure 5 depends on a K-Means setup whose initialization is not fully documented, so the central clustering result is more fragile than the paper suggests.

The second weakness is the mismatch between the written methods and the code. The paper describes min-max normalization and inclusion of an encoded region variable. The clustering code does neither. Instead, it uses five numeric variables only and applies `StandardScaler()`. Min-max scaling appears later only for visualizing cluster averages. A reader following the paper text alone would rebuild the wrong pipeline.

Sample reporting is another problem. The paper refers to more countries than are actually used in the clustering stage. The released CSV has 167 rows, but the effective sample is 143 after dropping missing progress values, and the published cluster sizes also sum to 143. This should have been reported clearly because it affects the cluster solution and all later analyses.

The weakest section is the statistical validation. In `Codes.py`, the ANOVA and MANOVA tables are created from manually typed dictionaries rather than from computed outputs, and the comments describe them as example values. This means the reported statistical evidence for cluster separation is not independently verifiable from the repository.

The classification section is also weaker than it first appears. The repository contains several overlapping evaluation blocks, and the Random Forest setup is not internally consistent across them. One section uses a standalone Random Forest on an unstratified split for feature importance, while later sections use scaled inputs and stratified sampling in a multi-model comparison. This makes it unclear which specification corresponds to the published results and helps explain why the Random Forest row still does not match cleanly.

There is also a broader design issue. The supervised models are trained to predict cluster labels that were themselves created from the same input variables. High classifier accuracy therefore shows internal separability, not strong external validation of real sustainability groupings. Overall, the study is more convincing as an exploratory exercise than as a fully reproducible AI application. The five-cluster result is broadly plausible, but the release would be much stronger with a single clean script, explicit hyperparameters, pinned package versions, consistent sample reporting, and programmatically generated tables.
