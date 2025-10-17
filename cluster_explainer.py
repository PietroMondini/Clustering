import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
import statsmodels.stats.multitest as smm
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. Generate toy dataset (patients)
# ---------------------------
np.random.seed(0)

# continuous variables
age = np.random.normal(60, 10, 200)
bmi = np.random.normal(27, 5, 200)
risk_score = np.random.uniform(0, 1, 200)

# binary variables (drugs/diseases)
diabetes = np.random.binomial(1, 0.3, 200)
hypertension = np.random.binomial(1, 0.4, 200)
statin = np.random.binomial(1, 0.25, 200)

X = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'risk_score': risk_score,
    'diabetes': diabetes,
    'hypertension': hypertension,
    'statin': statin
})

# ---------------------------
# 2. Clustering (k-means)
# ---------------------------
km = KMeans(n_clusters=3, random_state=0)
X['cluster'] = km.fit_predict(X)

# ---------------------------
# 3. Summaries
# ---------------------------
summary = X.groupby('cluster').agg(['mean','std','count'])
print("\nCluster summaries (mean/std):")
print(summary)

# ---------------------------
# 4. Statistical tests
# ---------------------------
pvals = {}
for col in ['age','bmi','risk_score']:
    groups = [X.loc[X.cluster==k, col] for k in X.cluster.unique()]
    pvals[col] = stats.kruskal(*groups).pvalue

for col in ['diabetes','hypertension','statin']:
    contingency = pd.crosstab(X[col], X['cluster'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency, correction=False)
    pvals[col] = p

# Multiple testing correction
_, pvals_fdr, _, _ = smm.multipletests(list(pvals.values()), method='fdr_bh')
pvals_corr = dict(zip(pvals.keys(), pvals_fdr))

print("\nCorrected p-values (FDR):")
print(pvals_corr)

# ---------------------------
# 5. Random Forest importance
# ---------------------------
X_feats = X.drop(columns='cluster')
y = X['cluster']
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_feats, y)

perm = permutation_importance(rf, X_feats, y, n_repeats=10, random_state=0)
importances = pd.Series(perm.importances_mean, index=X_feats.columns).sort_values(ascending=False)

print("\nPermutation Importances:")
print(importances)

# ---------------------------
# 6. Visualization
# ---------------------------
# Heatmap of cluster means (continuous and binary features)
plt.figure(figsize=(8,5))
cluster_means = X.groupby('cluster').mean()
sns.heatmap(cluster_means, annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title("Heatmap of Cluster Centroids (mean feature values)")
plt.ylabel("Cluster")
plt.xlabel("Feature")
plt.show()

# Boxplots for continuous variables
for col in ['age', 'bmi', 'risk_score']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='cluster', y=col, data=X)
    plt.title(f"{col.capitalize()} by cluster")
    plt.show()

# Barplots for binary features
for col in ['diabetes', 'hypertension', 'statin']:
    plt.figure(figsize=(6,4))
    sns.barplot(x='cluster', y=col, data=X, estimator=np.mean)
    plt.title(f"Proportion with {col} by cluster")
    plt.show()
