from datasets import DataLoader
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt

class Visualiser:
    def __init__(self, dl: DataLoader, config) -> None:
        self.dl = dl
        self.cont_cols = self.dl.test.columns
        dl.train.describe()
        self.colors = config["colors"]


    def featDistribution(self):
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

    def featuresViolinPlot(self):
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
    def __init__(self) -> None:
        pass


# fillna
# impute
# fill as missing

class Imputation:
    def __init__(self) -> None:
        pass


# Outliers
# Binning values