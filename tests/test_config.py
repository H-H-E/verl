# tests/test_config.py
import pytest
import yaml # For creating temp YAML
from pathlib import Path
from verl.atropos_config import load_atropos_config, AtroposConfig

def test_load_minimal_config(tmp_path: Path):
    '''Tests loading a minimal YAML configuration and checks default values.'''
    yaml_content = {
        "model": "dummy-model",
        "environments": ["gsm8k"]
    }
    config_file = tmp_path / "minimal_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    config = load_atropos_config(config_file)

    assert isinstance(config, AtroposConfig)
    assert config.model == "dummy-model"
    assert config.environments == ["gsm8k"]
    
    # Check for default values
    assert config.rollout_server_port == 8000
    assert config.use_sglang is False
    assert config.batch_size == 12 
    assert config.lr == 5e-6
    assert config.clip_ratio == 0.2
    assert config.ppo_epochs == 4
    assert config.kl_coef == 0.1
    assert config.entropy_coef == 0.01
    assert config.num_iterations == 1000
    assert config.reference_model is None
    assert config.tensor_parallel == 1
    assert config.inference_api_port == 8001


def test_load_override_defaults(tmp_path: Path):
    '''Tests loading a YAML that overrides some default values.'''
    yaml_content = {
        "model": "override-model",
        "environments": ["env1", "env2"],
        "rollout_server_port": 9000,
        "use_sglang": True,
        "lr": 1e-5,
        "num_iterations": 50
    }
    config_file = tmp_path / "override_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    config = load_atropos_config(config_file)

    assert config.model == "override-model"
    assert config.environments == ["env1", "env2"]
    assert config.rollout_server_port == 9000
    assert config.use_sglang is True
    assert config.lr == 1e-5
    assert config.num_iterations == 50
    
    # Check that other defaults are still applied
    assert config.batch_size == 12
    assert config.clip_ratio == 0.2


def test_load_unknown_fields_ignored(tmp_path: Path):
    '''Tests that unknown fields in YAML are ignored and defaults are applied.'''
    yaml_content = {
        "model": "test-model",
        "environments": ["test-env"],
        "unknown_field_1": "some_value",
        "another_unknown_param": 12345
    }
    config_file = tmp_path / "unknown_fields_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    config = load_atropos_config(config_file)

    assert config.model == "test-model"
    assert config.environments == ["test-env"]
    assert config.rollout_server_port == 8000 # Default
    
    # Assert that unknown fields are not attributes of the config object
    assert not hasattr(config, "unknown_field_1")
    assert not hasattr(config, "another_unknown_param")
