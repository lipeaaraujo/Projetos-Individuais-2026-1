import warnings

import pytest
from starlette.testclient import TestClient

from src import db
from src.config import Settings
from src.contract import PreviaOperacional

warnings.filterwarnings("ignore")


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "api.db")
    import src.api.main as api
    monkeypatch.setattr(api, "get_settings", lambda: Settings(db_path=path))
    db.init_db(path)
    seed = [
        ("Direcional", 2025, 1, 1_000_000, 5000),
        ("Direcional", 2026, 1, 1_500_000, 6000),
        ("Trisul", 2026, 1, 300_000, 1200),
    ]
    with db.connect(path) as conn:
        for emp, ano, tri, vgv, un in seed:
            did = db.register_document(
                conn, pdf_url=f"http://x/{emp}{ano}{tri}.pdf", source_page="http://ri",
                url_hash=f"{emp}{ano}{tri}", empresa=emp,
            )
            db.mark_downloaded(conn, did, sha256=f"s{emp}{ano}{tri}", file_path="x")
            p = PreviaOperacional.model_validate(
                {"empresa": emp, "ano": ano, "trimestre": tri,
                 "lancamentos": {"vgv_total_mil": vgv, "unidades": un}}
            )
            db.persist_extraction(conn, did, p)
            db.mark_processed(conn, did)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_conjuntura_filters(client):
    r = client.get("/api/conjuntura", params={"empresa": "direcional", "ano": 2026, "trimestre": 1, "tipo": "lancamentos"}).json()
    assert r["count"] == 1
    row = r["resultados"][0]
    assert row["vgv_total_mil"] == 1_500_000
    assert row["lineage"]["pdf_url"].endswith(".pdf")  # lineage exposed


def test_conjuntura_empresa_isolation(client):
    r = client.get("/api/conjuntura", params={"empresa": "trisul"}).json()
    assert {row["empresa"] for row in r["resultados"]} == {"Trisul"}


def test_series_computes_yoy_from_absolutes(client):
    r = client.get("/api/series", params={"empresa": "direcional", "tipo": "lancamentos"}).json()
    last = r["serie"][-1]
    assert last["periodo"] == "1T26"
    # (1.5M - 1.0M)/1.0M = +50% computed by us, not taken from marketing
    assert last["vgv_total_yoy_pct"] == 50.0
    assert last["unidades_yoy_pct"] == 20.0


def test_documents_lineage(client):
    r = client.get("/api/documents").json()
    assert r["count"] == 3
    assert all(d["status"] == "processed" for d in r["documentos"])
