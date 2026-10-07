import os
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logger import logger

def initialize_project_directories():
    """Sets up all dataset, model, upload, and evaluation directories."""
    config = get_config()
    classes = config["classes"]
    paths = config["paths"]

    splits = ["train", "validation", "test"]
    dataset_root = Path(paths["dataset_root"])

    for split in splits:
        for cls in classes:
            split_dir = dataset_root / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            gitkeep = split_dir / ".gitkeep"
            if not any(split_dir.iterdir()):
                gitkeep.touch()

    # Create additional operational directories
    for path_key in ["model_dir", "evaluation_dir", "uploads_dir"]:
        p = Path(paths[path_key])
        p.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path(paths["model_dir"]) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    test_results_dir = Path(paths["evaluation_dir"]) / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    failure_cases_dir = Path(paths["evaluation_dir"]) / "failure_cases"
    failure_cases_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project directory structure initialized successfully.")

if __name__ == "__main__":
    initialize_project_directories()
