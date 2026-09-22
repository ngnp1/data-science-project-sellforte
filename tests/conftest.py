from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = PROJECT_ROOT / "synthetic_data_generator"
VENV_PYTHON = Path(sys.executable)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def generator_dir() -> Path:
    return GENERATOR_DIR


@pytest.fixture
def venv_python() -> Path:
    return VENV_PYTHON


def pytest_collection_modifyitems(items):
    import shutil
    datasets = PROJECT_ROOT / "benchmark/datasets"
    have_data = all((datasets / split / f"{split}_001/media.csv").is_file()
                    for split in ("dev", "test"))
    for item in items:
        if item.get_closest_marker("benchmark_data") and not have_data:
            item.add_marker(pytest.mark.skip(reason="Generated benchmark datasets are not installed"))
        if item.get_closest_marker("requires_r") and not shutil.which("Rscript"):
            item.add_marker(pytest.mark.skip(reason="Rscript is not installed"))
