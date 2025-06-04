# tests/test_config_type_errors.py
import pytest
import yaml
from pathlib import Path
from verl.atropos_config import load_atropos_config

def test_missing_model_field(tmp_path: Path):
    '''Tests that ValueError is raised if 'model' field is missing.'''
    yaml_content = {
        "environments": ["gsm8k"]
        # model is missing
    }
    config_file = tmp_path / "missing_model.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    with pytest.raises(ValueError, match="Configuration error: 'model' field is required"):
        load_atropos_config(config_file)

def test_missing_environments_field(tmp_path: Path):
    '''Tests that ValueError is raised if 'environments' field is missing.'''
    yaml_content = {
        "model": "dummy-model"
        # environments is missing
    }
    config_file = tmp_path / "missing_environments.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    with pytest.raises(ValueError, match="Configuration error: 'environments' list is required"):
        load_atropos_config(config_file)

def test_empty_environments_list(tmp_path: Path):
    '''Tests that ValueError is raised if 'environments' list is empty.'''
    yaml_content = {
        "model": "dummy-model",
        "environments": [] # empty list
    }
    config_file = tmp_path / "empty_environments.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    with pytest.raises(ValueError, match="Configuration error: 'environments' list is required"):
        load_atropos_config(config_file)

def test_empty_yaml_file(tmp_path: Path):
    '''Tests that ValueError is raised for an empty YAML file (missing model and environments).'''
    config_file = tmp_path / "empty.yaml"
    with open(config_file, 'w') as f:
        f.write("") # Empty file

    # Depending on __post_init__ order, it might catch model or environments first.
    # Let's catch the general case of ValueError.
    with pytest.raises(ValueError, match="Configuration error: 'model' field is required"):
        load_atropos_config(config_file)


def test_config_file_not_found():
    '''Tests that FileNotFoundError is raised if the config file doesn't exist.'''
    with pytest.raises(FileNotFoundError):
        load_atropos_config("non_existent_config.yaml")

def test_invalid_yaml_format(tmp_path: Path):
    '''Tests that ValueError is raised for a malformed YAML file.'''
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, 'w') as f:
        f.write("model: dummy-model\nenvironments: [gsm8k") # Malformed YAML (missing closing bracket)

    with pytest.raises(ValueError, match="Error parsing YAML file"):
        load_atropos_config(config_file)

def test_type_mismatch_field(tmp_path: Path):
    '''Tests that ValueError or TypeError is raised for type mismatches.'''
    yaml_content = {
        "model": "dummy-model",
        "environments": ["gsm8k"],
        "rollout_server_port": "not_an_integer" # Type mismatch
    }
    config_file = tmp_path / "type_mismatch.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    # The dataclass itself might raise a TypeError if strict type checking is enforced
    # or our load_atropos_config might catch it if it tries to coerce.
    # The current implementation of load_atropos_config tries to instantiate
    # AtroposConfig(**filtered_yaml_data), which would lead to a TypeError
    # if rollout_server_port="not_an_integer" is passed.
    # This TypeError is caught and re-raised as a ValueError.
    with pytest.raises(ValueError, match="Error creating AtroposConfig from YAML data"):
        load_atropos_config(config_file)
