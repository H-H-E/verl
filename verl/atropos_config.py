from dataclasses import dataclass, field, fields
from typing import List, Optional, Union
from pathlib import Path
import yaml # Add PyYAML import

@dataclass
class AtroposConfig:
    # Required fields (no defaults in terms of user input)
    environments: List[str] = field(default_factory=list)
    model: str = ""

    # Fields with defaults
    rollout_server_port: int = 8000
    inference_api_port: int = 8001
    use_sglang: bool = False
    reference_model: Optional[str] = None
    batch_size: int = 12
    tensor_parallel: int = 1

    # VeRL-PPO hyperparams (with defaults)
    lr: float = 5e-6
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    kl_coef: float = 0.1
    entropy_coef: float = 0.01

    num_iterations: int = 1000

    def __post_init__(self):
        # Basic validation after loading.
        # More specific validation (e.g. model format) can be added if needed.
        if not self.model: # Check if model string is empty
            raise ValueError("Configuration error: 'model' field is required and cannot be empty.")
        if not self.environments: # Check if environments list is empty
            raise ValueError("Configuration error: 'environments' list is required and cannot be empty.")


def load_atropos_config(config_path: Union[str, Path]) -> AtroposConfig:
    '''Loads Atropos configuration from a YAML file, validates, and returns an AtroposConfig object.'''

    # Ensure config_path is a Path object
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML data from the file
    try:
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data is None: # Handle empty YAML file
            yaml_data = {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {config_path}\n{e}")

    # Create a default config instance
    # The dataclass defaults will be used for any keys not in yaml_data

    # Get all field names from the dataclass
    config_field_names = {f.name for f in fields(AtroposConfig)}

    # Filter yaml_data to only include keys that are actual fields in AtroposConfig
    filtered_yaml_data = {k: v for k, v in yaml_data.items() if k in config_field_names}

    try:
        # Instantiate AtroposConfig with values from YAML, falling back to defaults
        config = AtroposConfig(**filtered_yaml_data)
        # The __post_init__ will handle the validation for 'model' and 'environments'
    except TypeError as e:
        # This can happen if YAML contains keys not in AtroposConfig and not filtered,
        # or if there's a type mismatch that dataclasses can't handle directly.
        # Our filtering should prevent unknown keys, but type mismatches might still occur.
        raise ValueError(f"Error creating AtroposConfig from YAML data: {e}")

    return config
