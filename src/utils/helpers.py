"""
Helper Utilities
Common utility functions
"""

import json
from datetime import datetime

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
