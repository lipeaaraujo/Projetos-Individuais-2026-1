"""Pipeline idempotency: the same PDF content is extracted at most once."""

from pathlib import Path

import pytest

from src import db, pipeline
from src.config import Settings
from src.contract import PreviaOperacional
from src.extraction.chunking import ChunkPlan
from src.ingestion.adapters.base import DocumentRef

PDF = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "projeto-individual-4" / "exemplo_Boletim_Conjuntura_2025_3T.pdf"
)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        db_path=str(tmp_path / "t.db"),
        pdf_dir=str(tmp_path / "pdfs"),
        gemini_api_key="fake",
    )


@pytest.fixture
def fake_extract(monkeypatch):
    calls = {"n": 0}

    def _fake(pdf_bytes, *, api_key, model):
        calls["n"] += 1
        previa = PreviaOperacional.model_validate(
            {"empresa": "MRV", "ano": 2025, "trimestre": 3,
             "lancamentos": {"vgv_total_mil": 1000, "unidades": 100}}
        )
        plan = ChunkPlan("full-scan", pdf_bytes, 1, 1, "test")
        return previa, plan

    monkeypatch.setattr(pipeline, "extract", _fake)
    return calls


def _ref():
    return DocumentRef(pdf_url=PDF.as_uri(), source_page="local", empresa="MRV")


@pytest.mark.skipif(not PDF.exists(), reason="example PDF not present")
def test_first_ingest_processes(settings, fake_extract):
    Path(settings.pdf_dir).mkdir(parents=True, exist_ok=True)
    res = pipeline.ingest_ref(_ref(), settings)
    assert res.status == "processed"
    assert fake_extract["n"] == 1


@pytest.mark.skipif(not PDF.exists(), reason="example PDF not present")
def test_second_ingest_skipped_no_llm_call(settings, fake_extract):
    Path(settings.pdf_dir).mkdir(parents=True, exist_ok=True)
    pipeline.ingest_ref(_ref(), settings)
    res = pipeline.ingest_ref(_ref(), settings)  # identical content
    assert res.status == "skipped"
    assert fake_extract["n"] == 1  # LLM was NOT called again -> cost saved

    with db.connect(settings.db_path) as conn:
        n = conn.execute("SELECT COUNT(*) c FROM metrics").fetchone()["c"]
    assert n == 2  # still just one quarter's two blocks
