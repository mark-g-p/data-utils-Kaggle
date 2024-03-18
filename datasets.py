import pandas as pd
import json
import os

# Function to load configuration from a JSON file
def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

class DataLoader():
    def __init__(self, config):
        self.train             = pd.read_csv(os.path.join(config.path,"train.csv"), index_col = 'id')
        self.test              = pd.read_csv(os.path.join(config.path ,"test.csv"), index_col = 'id')
        self.targets           = config.targets 
        self.conjoin_orig_data = config.conjoin_orig_data

        if self.conjoin_orig_data == "Y":
            self.original      = pd.read_csv(config.orig_path, index_col = "id")
        else:
            self.original      = self.train
        
        self.sub_fl   = pd.read_csv(os.path.join(config.path, "sample_submission.csv"))
        # Remove non-standard symbols from column names
        for tbl in [self.train, self.original, self.test]:
            tbl.columns = tbl.columns.str.replace(r"\(|\)|\s+","", regex = True)
    
    def _AddSourceCol(self):
# Add column with source of data. Set list of feature columns names
        self.train['Source']    = "Competition"
        self.test['Source']     = "Competition"
        self.original['Source'] = "Original"

        self.feat_list = self.test.columns
        return self