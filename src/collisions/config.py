from typing import Any, Dict, List

import yaml
from pydantic import BaseModel

class Config(BaseModel):
    """Project Configuration"""

    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load config from YAML file"""
        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
