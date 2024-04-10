from datasets import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.stats import shapiro, normaltest
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier


class Visualiser:
    def __init__(self, dl: DataLoader, config) -> None:
        self.dl = dl
        self.cont_cols = self.dl.test.columns
        dl.train.describe()
        self.colors = config["colors"]

    def feat_distribution(self) -> None:
        num_plots = len(self.cont_cols)
        num_cols = 3
        num_rows = -(-num_plots // num_cols)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(21, 5 * num_rows)
        )  # Adjust the figure size as needed

        for i, feature in enumerate(self.cont_cols):
            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col] if num_rows > 1 else axes[col]

            sns.histplot(
                self.dl.train[feature],
                color=self.colors[0],
                label="Train",
                alpha=0.5,
                bins=30,
                ax=ax,
            )
            sns.histplot(
                self.dl.test[feature],
                color=self.colors[1],
                label="Test",
                alpha=0.5,
                bins=30,
                ax=ax,
            )
            sns.histplot(
                self.dl.original[feature],
                color=self.colors[2],
                label="Original",
                alpha=0.5,
                bins=30,
                ax=ax,
            )

            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.legend()

        if num_plots % num_cols != 0:
            for j in range(num_plots % num_cols, num_cols):
                axes[-1, j].axis("off")

        plt.tight_layout()
        plt.show()

    def features_violin_plot(self) -> None:
        num_plots = len(self.cont_cols)
        num_cols = 2
        num_rows = -(-num_plots // num_cols)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(21, 5 * num_rows)
        )  # Adjust the figure size as needed

        for i, feature in enumerate(self.cont_cols):
            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col] if num_rows > 1 else axes[col]

            sns.violinplot(
                x=self.dl.train[feature],
                color=self.colors[0],
                label="Train",
                alpha=0.5,
                ax=ax,
            )
            sns.violinplot(
                x=self.dl.test[feature],
                color=self.colors[1],
                label="Test",
                alpha=0.5,
                ax=ax,
            )
            sns.violinplot(
                x=self.dl.original[feature],
                color=self.colors[2],
                label="Original",
                alpha=0.5,
                ax=ax,
            )

            ax.legend()

            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.legend()

        if num_plots % num_cols != 0:
            for j in range(num_plots % num_cols, num_cols):
                axes[-1, j].axis("off")

        plt.tight_layout()
        plt.show()


class StatisticalTests:
    def __init__(self, dl: DataLoader, config) -> None:
        self.dl = dl
        self.cont_cols = self.dl.test.columns

    def test_normality(self) -> None:
        for df, name in zip(
            [self.dl.train, self.dl.test, self.dl.original],
            ["Train", "Test", "Original"],
        ):
            # Select only numeric columns
            print(f"\n{'-'*10}Testing features from {name}{'-'*10}\n")
            numeric_cols = df.select_dtypes(include=["float"]).columns
            normal_columns = []

            for col in numeric_cols:
                # Check normality using Shapiro-Wilk test if #samples <5000, ignore null values
                # For larger samples use D’Agostino-Pearson Test as S-W test is too sensitive
                stat, p = 0, 0
                if df.shape[0] < 5000:
                    stat, p = shapiro(df[col].dropna())

                else:
                    stat, p = normaltest(df[col], nan_policy="omit")

                print(f"\nColumn: {col}")
                print(f"Statistics={stat}, p={p}")
                if p > 0.05:
                    print(f"{col} looks Gaussian (fail to reject H0)")
                    normal_columns.append(col)
                else:
                    print(f"{col} does not look Gaussian (reject H0)")

            if len(normal_columns) > 0:
                print(normal_columns)
            else:
                print("No features passed the Normality test")


# fillna
# impute
# fill as missing


class Imputation:
    def __init__(self) -> None:
        pass


def adversarial_validation(train, test, target="target"):
    """Do adversarial validation for chosen datasets.
    Based on https://www.kaggle.com/code/kevinbonnes/adversarial-validation/notebook
    """
    # Create a 'source' feature that is 1 for 1st dataset instances and 0 for 2nd instances

    train["source"] = 1
    test["source"] = 0
    feat_to_remove = ["source"]
    if isinstance(target, str):
        feat_to_remove.append(target)
    elif isinstance(target, list):
        feat_to_remove += target

    df = pd.concat([train, test], axis=0)
    X = df.drop(columns=feat_to_remove, errors="ignore")
    y = df["source"]

    # Identify categorical features, fortunately Catboost doesn't need to encode them
    categorical_features_indices = np.where(X.dtypes == "object")[0]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    clf = CatBoostClassifier(cat_features=categorical_features_indices, verbose=0)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc_score = auc(fpr, tpr)

    print(f"AUC: {auc_score:.2f}")

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC: {auc_score:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Remove the 'source' column from the original datasets
    train.drop("source", axis=1, inplace=True)
    test.drop("source", axis=1, inplace=True)


def plot_importances(model, holdout_data, features):
    shap_values = model.get_feature_importance(holdout_data, type="ShapValues")
    shap_values = shap_values[:, :-1]
    shap.summary_plot(
        shap_values, holdout_data, feature_names=features, plot_type="bar"
    )
