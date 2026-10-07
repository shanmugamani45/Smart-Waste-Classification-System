import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Automatically discover project root (two levels up from src/utils)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigLoader:
    """Singleton and factory for loading and validating application YAML configuration."""

    _cached_config: Dict[str, Any] = {}

    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        """Loads configuration from YAML file, resolving relative paths against PROJECT_ROOT."""
        target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

        if not target_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cls._validate_schema(config)
        cls._resolve_paths(config)
        cls._cached_config = config
        return config

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Returns the loaded configuration, or loads default if not yet loaded."""
        if not cls._cached_config:
            return cls.load_config()
        return cls._cached_config

    @staticmethod
    def _validate_schema(cfg: Dict[str, Any]) -> None:
        """Validates that all essential sections and fields exist."""
        required_sections = ["model", "classes", "training", "inference", "circular_rules", "paths", "api"]
        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing mandatory configuration section: '{section}'")

        if not isinstance(cfg["classes"], list) or len(cfg["classes"]) == 0:
            raise ValueError("Configuration 'classes' must be a non-empty list of class strings.")

        if cfg["model"]["num_classes"] != len(cfg["classes"]):
            raise ValueError(
                f"Config mismatch: model.num_classes={cfg['model']['num_classes']} "
                f"does not match classes count={len(cfg['classes'])}"
            )

    @staticmethod
    def _resolve_paths(cfg: Dict[str, Any]) -> None:
        """Converts relative paths in cfg['paths'] to absolute Paths."""
        paths_section = cfg.get("paths", {})
        resolved = {}
        for key, val in paths_section.items():
            path_obj = Path(val)
            if not path_obj.is_absolute():
                resolved[key] = str((PROJECT_ROOT / path_obj).resolve())
            else:
                resolved[key] = str(path_obj.resolve())
        cfg["paths"] = resolved


def get_config() -> Dict[str, Any]:
    """Convenience getter for the global configuration."""
    return ConfigLoader.get_config()
