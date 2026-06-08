"""Pipeline orchestration: discover -> download -> hash -> extract -> persist.

Idempotency is enforced twice: by URL before download (in the poller) and by
SHA-256 content hash before the LLM call here, so no PDF is ever sent to Gemini
more than once. Every persisted metric is linked to its source document, giving
full data lineage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import db
from .config import Settings
from .extraction.extractor import extract
from .ingestion.adapters.base import DocumentRef, RIAdapter
from .ingestion.poller import discover_new, url_hash


@dataclass
class IngestResult:
    pdf_url: str
    status: str           # processed | skipped | failed
    detail: str = ""


def _download(ref: DocumentRef, settings: Settings) -> bytes:
    if ref.pdf_url.startswith("file://"):
        from urllib.parse import urlparse, unquote

        return Path(unquote(urlparse(ref.pdf_url).path)).read_bytes()
    r = httpx.get(
        ref.pdf_url,
        headers={"User-Agent": settings.user_agent},
        timeout=settings.http_timeout,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.content


def ingest_ref(ref: DocumentRef, settings: Settings) -> IngestResult:
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as conn:
        doc_id = db.register_document(
            conn,
            pdf_url=ref.pdf_url,
            source_page=ref.source_page,
            url_hash=url_hash(ref.pdf_url),
            empresa=ref.empresa,
            title=ref.title,
        )

    try:
        pdf_bytes = _download(ref, settings)
    except httpx.HTTPError as e:
        with db.connect(settings.db_path) as conn:
            db.mark_failed(conn, doc_id, f"download: {e}")
        return IngestResult(ref.pdf_url, "failed", str(e))

    sha = hashlib.sha256(pdf_bytes).hexdigest()
    with db.connect(settings.db_path) as conn:
        existing = db.find_by_sha256(conn, sha)
        if existing and existing["id"] != doc_id:
            if existing["status"] == "processed":
                db.mark_skipped(conn, doc_id, "conteúdo idêntico já processado")
                return IngestResult(ref.pdf_url, "skipped", "conteúdo já processado")
            # prior attempt for this content never finished: retry on its row,
            # discard the one we just registered (avoids sha256 UNIQUE clash).
            db.delete_document(conn, doc_id)
            doc_id = existing["id"]

        path = str(Path(settings.pdf_dir) / f"{sha[:16]}.pdf")
        Path(path).write_bytes(pdf_bytes)
        db.mark_downloaded(conn, doc_id, sha256=sha, file_path=path)

    if not settings.gemini_api_key:
        with db.connect(settings.db_path) as conn:
            db.mark_failed(conn, doc_id, "GEMINI_API_KEY ausente")
        return IngestResult(ref.pdf_url, "failed", "GEMINI_API_KEY ausente")

    try:
        previa, plan = extract(
            pdf_bytes, api_key=settings.gemini_api_key, model=settings.gemini_model
        )
    except Exception as e:
        with db.connect(settings.db_path) as conn:
            db.mark_failed(conn, doc_id, f"extração: {e}")
        return IngestResult(ref.pdf_url, "failed", str(e))

    with db.connect(settings.db_path) as conn:
        db.persist_extraction(conn, doc_id, previa)
        db.mark_processed(conn, doc_id)
    return IngestResult(
        ref.pdf_url,
        "processed",
        f"{previa.empresa} {previa.ano} {int(previa.trimestre)}T [{plan.strategy}]",
    )


def run_pipeline(adapters: list[RIAdapter], settings: Settings) -> list[IngestResult]:
    new = discover_new(adapters, settings.db_path)
    print(f"[pipeline] {len(new)} documento(s) novo(s) para processar.")
    results = []
    for ref in new:
        res = ingest_ref(ref, settings)
        print(f"  - {res.status:9} {res.detail or res.pdf_url[:70]}")
        results.append(res)
    return results
