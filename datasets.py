import pandas as pd
import json
import os

# Function to load configuration from a JSON file
def load_config(file_path : str):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

class DataLoader():
    def __init__(self, config):
        self.train             = pd.read_csv(os.path.join(config["path"],"train.csv"), index_col = 'id')
        self.test              = pd.read_csv(os.path.join(config["path"],"test.csv"), index_col = 'id')
        self.targets           = config["targets"]
        self.conjoin_orig_data = config["conjoin_orig_data"]

        if self.conjoin_orig_data == "Y":
            self.original      = pd.read_csv(config["orig_path"], index_col = "id")
        else:
            self.original      = self.train
        
        self.sub_fl   = pd.read_csv(os.path.join(config["path"], "sample_submission.csv"))
        # Standarize column names
        for tbl in [self.train, self.original, self.test]:
            tbl.columns = tbl.columns.str.replace(r"\(|\)|\s+","", regex = True)
            tbl.columns = map(str.lower, tbl.columns)
    
    def _addSourceCol(self):
# Add column with source of data. Set list of feature columns names
        self.train['Source']    = "Competition"
        self.test['Source']     = "Competition"
        self.original['Source'] = "Original"

        self.feat_list = self.test.columns
        return self
    
    def _displayDataHead(self, n=5):
        print(f"\nTrain set")
        print(self.train.head(n))
        print(f"\nTest set")
        print(self.test.head(n))
        if self.conjoin_orig_data == "Y":
            print(f"\nOriginal set")
            print(self.original.head(n))

    def dataDescription(self):
        print(f"\n{'-'*20} Information and descriptive statistics {'-'*20}\n")
        # Creating dataset information and description:
        for lbl, df in {'Train': self.train, 'Test': self.test, 'Original': self.original}.items():
            print(f"\n{lbl} description\n")
            print(df.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]))
            print(f"\n{lbl} information\n")
            print(df.info())
        return self
    
    def _ConjoinTrainOrig(self):
        if self.conjoin_orig_data == "Y":
            print(f"\n\nTrain shape before conjoining with original = {self.train.shape}")
            train = pd.concat([self.train, self.original], axis=0, ignore_index = True)
            print(f"Train shape after conjoining with original= {train.shape}")

            train = train.drop_duplicates(ignore_index=True)
            print(f"Train shape after de-duping = {train.shape}")

            train.index = range(len(train))
            train.index.name = 'id'

        else:
            print(f"\nWe are using the competition training data only")
            train = self.train
        return train
    
    def _missingData(self):
        # Display 
        for (df, name) in zip([self.train,self.test,self.original], ["Train", "Test", "Original"]):
            print(f"\n{'-'*20}Missing data {name}{'-'*20}\n")
            null_counts = df.isnull().sum()
            null_counts = null_counts[null_counts > 0]
            if null_counts.size > 0:
                print(null_counts)
            else:
                print(f"No null values found in {name} dataset.")
            
    def _uniqData(self):
        self._addSourceCol()
        for (df, name) in zip([self.train,self.test,self.original], ["Train", "Test", "Original"]):
            print(f"\n{'-'*20}Unique value count {name} {'-'*20}\n")
            print(df[self.feat_list].nunique())
