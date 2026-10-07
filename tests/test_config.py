import pytest
from pathlib import Path
from src.utils.config_loader import ConfigLoader, get_config

def test_config_loading():
    cfg = get_config()
    assert "model" in cfg
    assert "classes" in cfg
    assert "training" in cfg
    assert "inference" in cfg
    assert "circular_rules" in cfg
    assert "paths" in cfg
    assert "api" in cfg

def test_classes_mapping():
    cfg = get_config()
    expected_classes = ["e_waste", "glass", "metal", "organic", "paper", "plastic", "textile"]
    assert cfg["classes"] == expected_classes
    assert cfg["model"]["num_classes"] == 7

def test_circular_rules_exist_for_all_classes():
    cfg = get_config()
    rules = cfg["circular_rules"]
    for cls_name in cfg["classes"]:
        assert cls_name in rules, f"Missing circular rules for {cls_name}"
        assert "recommended_action" in rules[cls_name]
        assert "circular_strategy" in rules[cls_name]
        assert "environmental_priority" in rules[cls_name]
        assert rules[cls_name]["environmental_priority"] in ["LOW", "MEDIUM", "HIGH"]

def test_resolved_paths():
    cfg = get_config()
    paths = cfg["paths"]
    for path_key, path_val in paths.items():
        assert Path(path_val).is_absolute(), f"Path '{path_key}' must be absolute: {path_val}"
