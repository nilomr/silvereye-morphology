from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# get the repo root dir
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


# Import morphological data
df = pd.read_csv(DATA_DIR / "raw" / "morph-df.csv").drop(columns=["sample_name"])
df.columns = df.columns.str.lower()

# Drop rows with missing values
df = df.dropna()

# Get and plot the distribution of sample sizes of the target variable using
# seaborn, arrange the values in descending order
pop_counts = df["pop"].value_counts().sort_values(ascending=False)
sns.barplot(x=pop_counts.index, y=pop_counts)
# rotate labels
plt.xticks(rotation=45)
plt.show()


X = df.drop(columns=["pop"])
y = df["pop"]


# n_classes = len(df["pop"].unique())
# confusion_matrix_avg = np.zeros((n_classes -1, n_classes-1))

# for i in range(100):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
#     rf = RandomForestClassifier(n_estimators=1000)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     confusion_matrix_avg += confusion_matrix(y_test, y_pred, normalize="true")

# confusion_matrix_avg /= 100


# Train a random forest classifier with 'pop' as the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# plot a confusion matrix using seaborn with a dendrogram that shows the hierarchical
# clustering of the populations based on how often samples are misclassified

cm = confusion_matrix(y_test, y_pred, normalize="true")
g = sns.clustermap(
    cm, annot=True, col_cluster=False, xticklabels=rf.classes_, yticklabels=rf.classes_
)

bbox = g.ax_heatmap.get_position()
space = 0.15
g.ax_heatmap.set_position([bbox.x0 + space, bbox.y0, bbox.width, bbox.height])

# left align the y label text using set_yticklabels ha="left"
g.ax_heatmap.set_yticklabels(
    g.ax_heatmap.get_yticklabels(), rotation=0, ha="left", va="center"
)
# offset y labels by 10 pt
g.ax_heatmap.tick_params(axis="y", pad=90)

g.ax_heatmap.yaxis.tick_left()
plt.gcf().axes[-1].set_visible(False)
plt.show()


# get the feature importances and plot them using seaborn
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.show()
