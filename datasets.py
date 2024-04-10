import pandas as pd
import os
import numpy as np
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

    def _addSourceCol(self) -> None:
        # Add column with source of data. Set list of feature columns names
        self.train["Source"] = "Competition"
        self.test["Source"] = "Competition"
        self.original["Source"] = "Original"

        self.feat_list = self.test.columns

    def _displayDataHead(self, n=5) -> None:
        print(f"\nTrain set")
        print(self.train.head(n))
        print(f"\nTest set")
        print(self.test.head(n))
        if self.conjoin_orig_data == "Y":
            print(f"\nOriginal set")
            print(self.original.head(n))

    def dataDescription(self) -> None:
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

    def _conjoinTrainOrig(self) -> None:
        # Join two datasets and removes duplicates
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

    def _missingData(self) -> None:
        # Display
        for df, name in zip(
            [self.train, self.test, self.original], ["Train", "Test", "Original"]
        ):
            print(f"\n{'-'*20}Missing data {name}{'-'*20}\n")
            null_counts = df.isnull().sum()
            null_counts = null_counts[null_counts > 0]
            if null_counts.size > 0:
                # print(null_counts)
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

    def _uniqData(self) -> None:
        self._addSourceCol()
        for df, name in zip(
            [self.train, self.test, self.original], ["Train", "Test", "Original"]
        ):
            print(f"\n{'-'*20}Unique value count {name} {'-'*20}\n")
            print(df[self.feat_list].nunique())

    def _deleteMissing(self) -> None:
        # Never remove null values from test. It's expected to give results even with missing data.
        self.train = self.train.dropna()
        self.original = self.original.dropna()

    def _removeOutliers(self) -> None:
        # Never remove outliers from test. You are expected to use whole dataset.
        # Expects that there are no missing values in the datasets.
        try:
            # Fit the model
            clf = IsolationForest(contamination=0.01)
            clf.fit(self.train)

            # Predict the anomalies in the data
            pred = clf.predict(self.train)
            anomalies = self.train[pred == -1]
            self.train = self.train[pred != -1]
            print(f"{anomalies.shape[0]} outliers detected and removed.")
        except ValueError:
            print(
                "Missing values found. Before removing outliers make sure there are no NaNs or Nulls in the dataset."
            )
