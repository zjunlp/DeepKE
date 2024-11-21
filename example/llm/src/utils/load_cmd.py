import yaml

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
