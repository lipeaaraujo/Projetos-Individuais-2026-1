"""Data Catalog, structured store and lineage (SQLite).

- documents: ledger of every PDF seen. SHA-256 is the idempotency key; the row
  also records the RI page, canonical PDF URL and processing status.
- metrics: extracted absolute figures, one row per (document, kind). The FK to
  documents is the data lineage — any served number traces to its source PDF.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from .contract import MetricaOperacional, PreviaOperacional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    empresa       TEXT,                      -- best-effort, set by the adapter
    pdf_url       TEXT NOT NULL,             -- canonical URL of the PDF
    source_page   TEXT,                      -- RI page where the link was found
    sha256        TEXT UNIQUE,               -- content hash (idempotency key)
    url_hash      TEXT,                      -- hash of the URL (pre-download dedup)
    file_path     TEXT,                      -- local cached copy
    title         TEXT,                      -- link text / document title
    status        TEXT NOT NULL DEFAULT 'discovered',
                                             -- discovered|downloaded|processed|failed|skipped
    error         TEXT,                      -- last error message, if any
    detected_at   TEXT NOT NULL,
    processed_at  TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    empresa         TEXT NOT NULL,
    ano             INTEGER NOT NULL,
    trimestre       INTEGER NOT NULL,        -- 1..4
    tipo            TEXT NOT NULL,           -- 'lancamentos' | 'vendas'
    vgv_total_mil   REAL,                    -- absolute, R$ mil, NULL if absent
    vgv_co_mil      REAL,
    unidades        INTEGER,
    percentual_co   REAL,
    observacoes     TEXT,
    extracted_at    TEXT NOT NULL,
    -- one row per company/quarter/kind, latest extraction wins
    UNIQUE (empresa, ano, trimestre, tipo)
);

CREATE INDEX IF NOT EXISTS idx_metrics_filter
    ON metrics (empresa, ano, trimestre);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Yield a connection with FK enforcement and row access by column name."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create tables/indexes if they don't exist."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# --- Catalog / idempotency -----------------------------------------------


def find_by_sha256(conn: sqlite3.Connection, sha256: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
    ).fetchone()


def find_by_url_hash(conn: sqlite3.Connection, url_hash: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE url_hash = ?", (url_hash,)
    ).fetchone()


def register_document(
    conn: sqlite3.Connection,
    *,
    pdf_url: str,
    source_page: str,
    url_hash: str,
    empresa: Optional[str] = None,
    title: Optional[str] = None,
) -> int:
    """Insert a freshly *discovered* document (pre-download). Returns its id."""
    cur = conn.execute(
        """
        INSERT INTO documents (empresa, pdf_url, source_page, url_hash,
                               title, status, detected_at)
        VALUES (?, ?, ?, ?, ?, 'discovered', ?)
        """,
        (empresa, pdf_url, source_page, url_hash, title, _now()),
    )
    return int(cur.lastrowid)


def mark_downloaded(
    conn: sqlite3.Connection, doc_id: int, *, sha256: str, file_path: str
) -> None:
    conn.execute(
        "UPDATE documents SET sha256 = ?, file_path = ?, status = 'downloaded' "
        "WHERE id = ?",
        (sha256, file_path, doc_id),
    )


def mark_processed(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute(
        "UPDATE documents SET status = 'processed', processed_at = ? WHERE id = ?",
        (_now(), doc_id),
    )


def mark_failed(conn: sqlite3.Connection, doc_id: int, error: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'failed', error = ? WHERE id = ?",
        (error[:2000], doc_id),
    )


# --- Structured store -----------------------------------------------------


def _upsert_metric(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    empresa: str,
    ano: int,
    trimestre: int,
    tipo: str,
    metrica: MetricaOperacional,
    observacoes: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (document_id, empresa, ano, trimestre, tipo,
                             vgv_total_mil, vgv_co_mil, unidades, percentual_co,
                             observacoes, extracted_at)
        VALUES (:document_id, :empresa, :ano, :trimestre, :tipo,
                :vgv_total_mil, :vgv_co_mil, :unidades, :percentual_co,
                :observacoes, :extracted_at)
        ON CONFLICT(empresa, ano, trimestre, tipo) DO UPDATE SET
            document_id   = excluded.document_id,
            vgv_total_mil = excluded.vgv_total_mil,
            vgv_co_mil    = excluded.vgv_co_mil,
            unidades      = excluded.unidades,
            percentual_co = excluded.percentual_co,
            observacoes   = excluded.observacoes,
            extracted_at  = excluded.extracted_at
        """,
        {
            "document_id": document_id,
            "empresa": empresa,
            "ano": ano,
            "trimestre": trimestre,
            "tipo": tipo,
            "vgv_total_mil": metrica.vgv_total_mil,
            "vgv_co_mil": metrica.vgv_co_mil,
            "unidades": metrica.unidades,
            "percentual_co": metrica.percentual_co,
            "observacoes": observacoes,
            "extracted_at": _now(),
        },
    )


def persist_extraction(
    conn: sqlite3.Connection, document_id: int, previa: PreviaOperacional
) -> None:
    """Write the launches + sales blocks, both linked to document_id (lineage)."""
    for tipo, metrica in (("lancamentos", previa.lancamentos), ("vendas", previa.vendas)):
        _upsert_metric(
            conn,
            document_id=document_id,
            empresa=previa.empresa,
            ano=previa.ano,
            trimestre=int(previa.trimestre),
            tipo=tipo,
            metrica=metrica,
            observacoes=previa.observacoes,
        )
