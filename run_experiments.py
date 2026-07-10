import os
import json
import yaml
import sys
from src import config_manager

def load_yaml_config(config_path):
    """
    Load a YAML configuration file from the given path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is not a valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.full_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    return config

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py <yaml_config_path>")
        sys.exit(1)

    yaml_config_path = sys.argv[1]
    
    try:
        yaml_config = load_yaml_config(yaml_config_path)
        print("YAML Configurations loaded successfully")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading YAML config: {e}")

    configs, params = config_manager.parse_configs(yaml_config, '')
    
    for config, param in zip(configs, params):
        config['params_set'] = param
        config_manager.evauate_config(config)
        