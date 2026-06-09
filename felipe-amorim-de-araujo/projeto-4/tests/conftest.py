import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")
