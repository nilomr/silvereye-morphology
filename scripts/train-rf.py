# ──── IMPORTS ────────────────────────────────────────────────────────────────

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rasterio import pad
from scipy.cluster import hierarchy as sch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import permutation_test_score, train_test_split
from tqdm.autonotebook import tqdm

# ──── DATA INGEST ────────────────────────────────────────────────────────────


# get the repo root dir
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"

for d in [DATA_DIR, FIG_DIR]:
    if not d.exists():
        d.mkdir()


# Import morphological data
df = pd.read_csv(DATA_DIR / "raw" / "morph-df.csv")
df.columns = df.columns.str.lower()

# Drop unnecessary columns
df.drop(columns=["sample_name", "pop"], inplace=True)

# rename 'pop_state' to 'population'
df.rename(columns={"pop_state": "population"}, inplace=True)
# remove underscores from population names
df["population"] = df["population"].str.replace("_", " ")

# Drop rows with missing values
df = df.dropna()

# Remove populations with < 5 samples
pop_counts = df["population"].value_counts()
df = df[df["population"].isin(pop_counts[pop_counts >= 5].index)]


# ──── TRAIN THE CLASSIFIER ───────────────────────────────────────────────────

# To get around the class imbalance, we will resample the data with replacement
# so that each population has the same number of samples. We will then train a
# random forest classifier on the resampled data and use the test data to
# evaluate the model. We will repeat this process 100 times and average the
# confusion matrices to get a better estimate of the model's performance on
# random subsets.


n_classes = len(df["population"].unique())
confusion_matrix_avg = np.zeros((n_classes, n_classes))

nits = 100
feature_importances = np.zeros((nits, len(df.columns) - 1))

for i in tqdm(range(nits)):
    # Resample all classes to the same number of samples (15), with replacement
    X = (
        df.groupby("population")
        .apply(lambda x: x.sample(15, replace=True))
        .droplevel(0)
    )
    y = X["population"]
    X = X.drop(columns=["population"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=i
    )
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    # add the values to the corresponding indices in the larger array
    confusion_matrix_avg += cm
    feature_importances[i, :] = rf.feature_importances_

confusion_matrix_avg /= nits


# Now we train a random forest classifier with all the data in df

X = df.drop(columns=["population"])
y = df["population"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# get accuracy score
score = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {score:.2f}")


score, permutation_scores, pvalue = permutation_test_score(
    rf,
    X_test,
    y_test,
    scoring="accuracy",
    n_jobs=-1,
    n_permutations=200,
    random_state=42,
    verbose=2,
)

# print results of permutation test in the same line
print(f"Permutation test score: {score:.2f}, p-value: {pvalue:.5f}")


# now plot the distribution of the permutation scores and the actual score
sns.set_theme()
fig, ax = plt.subplots(figsize=(6, 3))
sns.histplot(
    permutation_scores,
    kde=True,
    stat="density",
    edgecolor="none",
    bins=15,
    label="Permutation scores",
)
plt.axvline(score, color="red", linestyle="--", label="Model accuracy")
plt.xlabel("Accuracy score")
plt.ylabel("Density")
plt.xlim(left=0)
plt.ylim(bottom=0, top=35)
plt.legend()
ax.legend(frameon=False, loc="upper left")
sns.despine()

# save plot as a pdf
plt.savefig(
    FIG_DIR / "permutation-test-score.pdf",
    bbox_inches="tight",
    transparent=True,
)


# plot the confusion matrix as a heatmap ordered by how often samples are
# misclassified between populations

cm = confusion_matrix_avg

# Perform hierarchical clustering on the confusion matrix
d = sch.distance.pdist(cm)
L = sch.linkage(d, method="complete")
ind = sch.fcluster(L, 0.5 * d.max(), "distance")

# Reorder the confusion matrix based on the clustering
cm_df = pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_)
new_index = cm_df.index[np.argsort(ind)]
cm_df = cm_df.reindex(index=new_index, columns=new_index)

# get the new order of the populations for the x and y tick labels
new_order_x = cm_df.columns
new_order_y = cm_df.index

fig, ax = plt.subplots(figsize=(8, 8))
g = sns.heatmap(
    cm_df,
    annot=True,
    fmt=".01f",
    xticklabels=new_order_x,
    yticklabels=new_order_y,
    linewidth=0.5,
    annot_kws={"fontsize": 8},
    square=True,  # set square parameter to True
)

# add predicted and true labels to the plot
g.set_xlabel("Predicted population")
g.set_ylabel("Actual population")
# remove colorbar
g.collections[0].colorbar.remove()

# save plot as a pdf
plt.savefig(
    FIG_DIR / "confusion-matrix.pdf",
    bbox_inches="tight",
    transparent=True,
)


# plot a confusion matrix using seaborn with a dendrogram that shows the hierarchical
# clustering of the populations based on how often samples are misclassified

cm = confusion_matrix_avg * 100
g = sns.clustermap(
    cm,
    annot=True,
    fmt=".0f",  # modify fmt parameter to remove percentage sign
    col_cluster=True,
    row_cluster=True,
    xticklabels=rf.classes_,
    yticklabels=rf.classes_,
    cmap="YlGnBu",
    tree_kws=dict(linewidths=1.5),
    annot_kws={"fontsize": 9},
)

bbox = g.ax_heatmap.get_position()
space = 0.18
g.ax_heatmap.set_position([bbox.x0 + space, bbox.y0, bbox.width, bbox.height])

# left align the y label text using set_yticklabels ha="left"
g.ax_heatmap.set_yticklabels(
    g.ax_heatmap.get_yticklabels(), rotation=0, ha="left", va="center"
)
# offset y labels by 10 pt
g.ax_heatmap.tick_params(axis="y", pad=100)
g.ax_heatmap.yaxis.tick_left()
plt.gcf().axes[-1].set_visible(False)

# remove the top dendrogram
g.ax_col_dendrogram.set_visible(False)
g.ax_heatmap.tick_params(left=False, bottom=False)

# Add predicted and true labels to the plot, at the top of the x axis and on the
# right of the y axis
g.ax_heatmap.set_xlabel("Predicted population", fontsize=12, labelpad=10)
g.ax_heatmap.xaxis.set_label_position("top")
g.ax_heatmap.set_ylabel("Actual population", fontsize=12)
g.ax_heatmap.yaxis.set_label_position("right")
# flip y axis label 180 degrees
g.ax_heatmap.set_ylabel(
    g.ax_heatmap.get_ylabel(), rotation=-90, fontsize=12, labelpad=20
)

# save plot as a pdf
plt.savefig(
    FIG_DIR / "confusion-matrix-clustermap.pdf",
    bbox_inches="tight",
    transparent=True,
)


# plot the feature importances as a categorical scatterplot with a bar at the
# mean
feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns)
feature_importances_df = feature_importances_df.melt(
    var_name="Feature", value_name="Importance"
)
feature_importances_df["Importance"] *= 100
feature_importances_df["Feature"] = (
    feature_importances_df["Feature"]
    .str.replace("_", " ")
    .str.replace("headlength", "head\nlength")
    .str.replace(" ", "\n")
    .str.replace("(", "\n(")
    .str.replace(")", "")
)

fig, ax = plt.subplots(figsize=(8, 8))
sns.stripplot(
    x="Importance",
    y="Feature",
    data=feature_importances_df,
    orient="h",
    color="black",
    size=3,
    ax=ax,
)
# add a bar at the mean of each feature
means = feature_importances_df.groupby("Feature").mean().reset_index()
ax.barh(means["Feature"], means["Importance"], color="#46848f", alpha=0.5, label="Mean")
ax.legend()
ax.set_xlabel("Importance (%)")
ax.set_ylabel("Feature")
sns.despine()


# save as a pdf
plt.savefig(
    FIG_DIR / "feature-importances.pdf",
    bbox_inches="tight",
    transparent=True,
)
