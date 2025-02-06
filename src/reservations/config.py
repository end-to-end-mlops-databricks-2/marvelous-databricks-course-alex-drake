from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Project Configuration"""

    data: str
    id_column: str
    num_features: List[str]
    cat_features: List[str]
    target: str
    features: List[str]
    catalog_name: str
    schema_name: str
    experiment_name: str
    parameters: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load config from YAML file"""

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str