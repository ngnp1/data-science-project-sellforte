"""The scaffolding test exists so Task 1 has a real gate: it proves the
fixtures resolve and that the R toolchain this whole plan depends on is
actually present."""
import pytest
import shutil
import subprocess


def test_fixtures_point_at_real_directories(project_root, generator_dir, venv_python):
    assert (project_root / "pytest.ini").is_file()
    assert (generator_dir / "generate_with_simmmulator.R").is_file()
    assert venv_python.is_file()


@pytest.mark.requires_r
def test_rscript_is_available_with_simmmulator():
    assert shutil.which("Rscript") is not None
    proc = subprocess.run(
        ["Rscript", "-e", 'cat("siMMMulator" %in% rownames(installed.packages()))'],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "TRUE"
