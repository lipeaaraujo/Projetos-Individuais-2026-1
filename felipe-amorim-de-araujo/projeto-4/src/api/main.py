"""Service layer: REST/JSON API over the structured store.

Endpoints:
  GET /api/conjuntura?empresa=&ano=&trimestre=  -> filtered absolute metrics
  GET /api/documents?empresa=                   -> catalog with lineage
  GET /health
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Query

from ..config import get_settings
from ..db import connect, init_db

app = FastAPI(
    title="API Conjuntura do Setor Habitacional",
    description="Dados operacionais absolutos de incorporadoras, extraídos por LLM.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    init_db(get_settings().db_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/conjuntura")
def conjuntura(
    empresa: Optional[str] = Query(None, description="Filtra por empresa (case-insensitive)"),
    ano: Optional[int] = Query(None),
    trimestre: Optional[int] = Query(None, ge=1, le=4),
    tipo: Optional[str] = Query(None, description="lancamentos | vendas"),
) -> dict:
    where, params = [], []
    if empresa:
        where.append("LOWER(m.empresa) = LOWER(?)")
        params.append(empresa)
    if ano is not None:
        where.append("m.ano = ?")
        params.append(ano)
    if trimestre is not None:
        where.append("m.trimestre = ?")
        params.append(trimestre)
    if tipo:
        where.append("m.tipo = ?")
        params.append(tipo)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT m.empresa, m.ano, m.trimestre, m.tipo,
               m.vgv_total_mil, m.vgv_co_mil, m.unidades, m.percentual_co,
               m.observacoes, m.extracted_at,
               d.id AS document_id, d.pdf_url, d.source_page
        FROM metrics m
        JOIN documents d ON d.id = m.document_id
        {clause}
        ORDER BY m.empresa, m.ano, m.trimestre, m.tipo
    """
    with connect(get_settings().db_path) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    for r in rows:
        r["lineage"] = {"document_id": r.pop("document_id"),
                        "pdf_url": r.pop("pdf_url"),
                        "source_page": r.pop("source_page")}
    return {"count": len(rows), "resultados": rows}


def _pct(curr, prev):
    if curr is None or prev in (None, 0):
        return None
    return round((curr - prev) / prev * 100, 1)


@app.get("/api/series")
def series(
    empresa: str = Query(..., description="Empresa obrigatória"),
    tipo: str = Query("lancamentos", description="lancamentos | vendas"),
) -> dict:
    """Série temporal com variações QoQ e YoY calculadas a partir dos valores
    ABSOLUTOS armazenados — não dos percentuais de marketing dos relatórios."""
    sql = """
        SELECT ano, trimestre, vgv_total_mil, vgv_co_mil, unidades
        FROM metrics
        WHERE LOWER(empresa) = LOWER(?) AND tipo = ?
        ORDER BY ano, trimestre
    """
    with connect(get_settings().db_path) as conn:
        rows = [dict(r) for r in conn.execute(sql, (empresa, tipo)).fetchall()]

    by_key = {(r["ano"], r["trimestre"]): r for r in rows}
    serie = []
    for r in rows:
        prev_q = by_key.get(
            (r["ano"] - 1, 4) if r["trimestre"] == 1 else (r["ano"], r["trimestre"] - 1)
        )
        prev_y = by_key.get((r["ano"] - 1, r["trimestre"]))
        serie.append({
            **r,
            "periodo": f"{r['trimestre']}T{str(r['ano'])[2:]}",
            "vgv_total_qoq_pct": _pct(r["vgv_total_mil"], prev_q and prev_q["vgv_total_mil"]),
            "vgv_total_yoy_pct": _pct(r["vgv_total_mil"], prev_y and prev_y["vgv_total_mil"]),
            "unidades_yoy_pct": _pct(r["unidades"], prev_y and prev_y["unidades"]),
        })
    return {"empresa": empresa, "tipo": tipo, "count": len(serie), "serie": serie}


@app.get("/api/documents")
def documents(empresa: Optional[str] = Query(None)) -> dict:
    where, params = "", []
    if empresa:
        where = "WHERE LOWER(empresa) = LOWER(?)"
        params.append(empresa)
    sql = f"""
        SELECT id, empresa, title, pdf_url, source_page, status, error,
               detected_at, processed_at
        FROM documents {where}
        ORDER BY detected_at DESC
    """
    with connect(get_settings().db_path) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    return {"count": len(rows), "documentos": rows}
