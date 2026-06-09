from src import db
from src.contract import PreviaOperacional


def _seed(conn, empresa="MRV", ano=2026, tri=1, tag=""):
    doc_id = db.register_document(
        conn, pdf_url=f"http://x/{empresa}{tag}.pdf", source_page="http://ri",
        url_hash=f"{empresa}{ano}{tri}{tag}", empresa=empresa, title="Prévia",
    )
    db.mark_downloaded(conn, doc_id, sha256=f"sha{empresa}{ano}{tri}{tag}", file_path="/tmp/x.pdf")
    p = PreviaOperacional.model_validate({
        "empresa": empresa, "ano": ano, "trimestre": tri,
        "lancamentos": {"vgv_total_mil": 2915000, "unidades": 10386},
        "vendas": {"vgv_co_mil": 2469000, "unidades": 9000},
    })
    db.persist_extraction(conn, doc_id, p)
    db.mark_processed(conn, doc_id)
    return doc_id


def test_persist_and_lineage(db_path):
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        doc_id = _seed(conn)
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT m.tipo, m.document_id, d.pdf_url, d.status "
            "FROM metrics m JOIN documents d ON d.id = m.document_id ORDER BY m.tipo"
        ).fetchall()
    assert {r["tipo"] for r in rows} == {"lancamentos", "vendas"}
    # every metric traces back to its source PDF (lineage)
    assert all(r["document_id"] == doc_id and r["pdf_url"].endswith(".pdf") for r in rows)
    assert all(r["status"] == "processed" for r in rows)


def test_idempotent_upsert_one_row_per_quarter(db_path):
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        _seed(conn, ano=2026, tri=1, tag="a")
        _seed(conn, ano=2026, tri=1, tag="b")  # distinct doc, same quarter -> upsert
    with db.connect(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM metrics WHERE empresa='MRV' AND ano=2026 AND trimestre=1"
        ).fetchone()["c"]
    assert n == 2  # exactly lancamentos + vendas, no duplicates


def test_find_by_sha256(db_path):
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        _seed(conn)
        assert db.find_by_sha256(conn, "shaMRV20261") is not None  # sha{empresa}{ano}{tri}{tag=''}
        assert db.find_by_sha256(conn, "missing") is None
