from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = PROJECT_ROOT / "synthetic_data_generator"
VENV_PYTHON = GENERATOR_DIR / ".venv" / "bin" / "python"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def generator_dir() -> Path:
    return GENERATOR_DIR


@pytest.fixture
def venv_python() -> Path:
    return VENV_PYTHON
