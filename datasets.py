import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


class DataLoader:
    def __init__(self, config) -> None:
        self.train = pd.read_csv(
            os.path.join(config["path"], "train.csv"), index_col="id"
        )
        self.test = pd.read_csv(
            os.path.join(config["path"], "test.csv"), index_col="id"
        )
        self.targets = config["targets"]
        self.conjoin_orig_data = config["conjoin_orig_data"]

        if self.conjoin_orig_data == "Y":
            self.original = pd.read_csv(config["orig_path"], index_col="id")
        else:
            self.original = self.train

        self.sub_fl = pd.read_csv(os.path.join(config["path"], "sample_submission.csv"))
        # Standarize column names
        for tbl in [self.train, self.original, self.test]:
            tbl.columns = tbl.columns.str.replace(r"\(|\)|\s+", "", regex=True)
            tbl.columns = map(str.lower, tbl.columns)

        self.delete_missing = config["delete_missing"]

    def _add_source_col(self) -> None:
        # Add column with source of data. Set list of feature columns names
        self.train["Source"] = "Competition"
        self.test["Source"] = "Competition"
        self.original["Source"] = "Original"

        self.feat_list = self.test.columns

    def _display_data_head(self, n=5) -> None:
        print(f"\nTrain set")
        print(self.train.head(n))
        print(f"\nTest set")
        print(self.test.head(n))
        if self.conjoin_orig_data == "Y":
            print(f"\nOriginal set")
            print(self.original.head(n))

    def data_description(self) -> None:
        print(f"\n{'-'*20} Information and descriptive statistics {'-'*20}\n")
        # Creating dataset information and description:
        for lbl, df in {
            "Train": self.train,
            "Test": self.test,
            "Original": self.original,
        }.items():
            print(f"\n{lbl} description\n")
            print(df.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]))
            print(f"\n{lbl} information\n")
            print(df.info())

    def _conjoin_train_orig(self) -> None:
        # Join two training dataset with original dataset and removes duplicates
        if self.conjoin_orig_data == "Y":
            print(
                f"\n\nTrain shape before conjoining with original = {self.train.shape}"
            )
            train = pd.concat([self.train, self.original], axis=0, ignore_index=True)
            print(f"Train shape after conjoining with original= {train.shape}")

            train.index = range(len(train))
            train.index.name = "id"

        else:
            print(f"\nWe are using the competition training data only")
            train = self.train

        train = train.drop_duplicates(ignore_index=True)
        print(f"Train shape after de-duping = {train.shape}")

    def _missing_data(self) -> None:
        # Display
        for df, name in zip(
            [self.train, self.test, self.original], ["Train", "Test", "Original"]
        ):
            print(f"\n{'-'*20}Missing data {name}{'-'*20}\n")
            null_counts = df.isnull().sum()
            null_counts = null_counts[null_counts > 0]
            if null_counts.size > 0:
                print(f"Incomplete rows: {null_counts.size}")
                print(
                    f"Percentage of all rows: {(null_counts.size/df.shape[0]*100):.3f}%"
                )
                if self.delete_missing == "Y":
                    print(f"\n{'-'*10}Dropping incomple rows from {name}{'-'*10}\n")
                    print(f"Dataframe size before removal {df.shape}")
                    df = df.dropna()
                    print(f"Dataframe size after removal {df.shape}")
            else:
                print(f"No null values found in {name} dataset.")

    def _uniq_data(self) -> None:
        self._add_source_col()
        for df, name in zip(
            [self.train, self.test, self.original], ["Train", "Test", "Original"]
        ):
            print(f"\n{'-'*20}Unique value count {name} {'-'*20}\n")
            print(df[self.feat_list].nunique())

    def _delete_missing(self) -> None:
        # Never remove null values from test. It's expected to give results even with missing data.
        self.train = self.train.dropna()
        self.original = self.original.dropna()

    def _remove_outliers(self) -> None:
        # Never remove outliers from test. You are expected to use whole dataset.
        # Expects that there are no missing values in the datasets.
        # Checks only numeric data. If you want to use it for categorical, encode them first.

        num_cols = self.train.select_dtypes(include=np.number).columns
        try:
            # Fit the model
            train_num = self.train[num_cols]
            clf = IsolationForest(contamination=0.01)
            clf.fit(train_num)

            # Predict the anomalies in the data
            pred = clf.predict(train_num)
            anomalies = train_num[pred == -1]
            self.train = train_num[pred != -1].dropna()
            print(f"{anomalies.shape[0]} outliers detected and removed.")

        except ValueError:
            print(
                "Missing values found. Before removing outliers make sure there are no NaNs or Nulls in the dataset."
            )

    def plot_distributions(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str = "DataFrame 1",
        name2: str = "DataFrame 2",
    ) -> None:
        """Plot histograms for common features between two dataframes.
        Doesn't keep original order of features.
        Histograms are scaled to sum up to 1 (i.e., to represent a probability distribution)
        to make visual comparison of two distributions easier."""
        numerical_columns1 = df1.select_dtypes(include=np.number).columns
        numerical_columns2 = df2.select_dtypes(include=np.number).columns
        numerical_columns = sorted(
            list(set(numerical_columns1) & set(numerical_columns2))
        )
        _, axs = plt.subplots(
            len(numerical_columns), figsize=(10, 5 * len(numerical_columns))
        )

        weights1 = np.ones(df1.shape[0]) / df1.shape[0]
        weights2 = np.ones(df2.shape[0]) / df2.shape[0]

        for i, col in enumerate(numerical_columns):

            axs[i].hist(
                df1[col],
                color="skyblue",
                edgecolor="black",
                alpha=0.5,
                label=f"{name1}",
                weights=weights1,
            )
            axs[i].hist(
                df2[col],
                color="red",
                edgecolor="black",
                alpha=0.5,
                label=f"{name2}",
                weights=weights2,
            )
            axs[i].set_title(f"Distributions of {col}")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
