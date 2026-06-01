"""
Central configuration loader for all scripts.

This module provides a centralized way to load path configurations from YAML,
eliminating scattered path definitions across individual scripts.
"""

from pathlib import Path
import yaml


def load_paths(config_path: str = "config/paths.yaml") -> dict:
    """
    Load path configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    dict
        Dictionary with all configured paths as Path objects.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file: {path}. "
            "Copy config/paths.yaml.example to config/paths.yaml and update the paths."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return {k: Path(v) if isinstance(v, str) else v for k, v in cfg.items()}
