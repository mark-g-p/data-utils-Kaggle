import json

# Function to load configuration from a JSON file. Probably make it as a different class
def load_config(file_path : str):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config
