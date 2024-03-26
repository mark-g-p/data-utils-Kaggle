from datasets import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, normaltest

class Visualiser:
    def __init__(self, dl: DataLoader, config) -> None:
        self.dl = dl
        self.cont_cols = self.dl.test.columns
        dl.train.describe()
        self.colors = config["colors"]


    def featDistribution(self) -> None:
        num_plots = len(self.cont_cols)
        num_cols = 3  
        num_rows = -(-num_plots // num_cols)  
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(21, 5 * num_rows))  # Adjust the figure size as needed

        for i, feature in enumerate(self.cont_cols):
            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            sns.histplot(self.dl.train[feature], color=self.colors[0], label='Train', alpha=0.5, bins=30, ax=ax)
            sns.histplot(self.dl.test[feature], color=self.colors[1], label='Test', alpha=0.5, bins=30, ax=ax)
            sns.histplot(self.dl.original[feature], color=self.colors[2], label='Original', alpha=0.5, bins=30, ax=ax)

            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()

        if num_plots % num_cols != 0:
            for j in range(num_plots % num_cols, num_cols):
                axes[-1, j].axis('off')

        plt.tight_layout()
        plt.show()

    def featuresViolinPlot(self) -> None:
        num_plots = len(self.cont_cols)
        num_cols = 2
        num_rows = -(-num_plots // num_cols)  
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(21, 5 * num_rows))  # Adjust the figure size as needed

        for i, feature in enumerate(self.cont_cols):
            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col] if num_rows > 1 else axes[col]
    
            sns.violinplot(x=self.dl.train[feature], color=self.colors[0], label="Train", alpha=0.5, ax=ax)
            sns.violinplot(x=self.dl.test[feature], color=self.colors[1], label="Test", alpha=0.5, ax=ax)
            sns.violinplot(x=self.dl.original[feature], color=self.colors[2], label="Original", alpha=0.5, ax=ax)
            
            ax.legend() 
            
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.legend()

        if num_plots % num_cols != 0:
            for j in range(num_plots % num_cols, num_cols):
                axes[-1, j].axis('off')

        plt.tight_layout()
        plt.show()
        
class StatisticalTests:
    def __init__(self, dl: DataLoader, config) -> None:
        self.dl = dl
        self.cont_cols = self.dl.test.columns

    def testNormality(self) -> None:
        for (df, name) in zip([self.dl.train, self.dl.test, self.dl.original], ["Train", "Test", "Original"]):
            # Select only numeric columns
            print(f"\n{'-'*10}Testing features from {name}{'-'*10}\n")
            numeric_cols = df.select_dtypes(include=['float']).columns
            normal_columns = []

            for col in numeric_cols:
                # Check normality using Shapiro-Wilk test if #samples <5000, ignore null values
                # For larger samples use D’Agostino-Pearson Test as S-W test is too sensitive
                stat, p = 0, 0 
                if df.shape[0] < 5000 :
                    stat, p = shapiro(df[col].dropna())
                    
                else:
                    stat, p = normaltest(df[col], nan_policy='omit')

                print(f"\nColumn: {col}")
                print(f"Statistics={stat}, p={p}")
                if p > 0.05:
                    print(f"{col} looks Gaussian (fail to reject H0)")
                    normal_columns.append(col)
                else:
                    print(f"{col} does not look Gaussian (reject H0)")

            if len(normal_columns)>0:
                print(normal_columns)
            else:
                print("No features passed the Normality test")

# fillna
# impute
# fill as missing

class Imputation:
    def __init__(self) -> None:
        pass


# Outliers
# Binning values