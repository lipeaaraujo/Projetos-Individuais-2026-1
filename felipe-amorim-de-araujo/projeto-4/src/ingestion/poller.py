"""Discovery + URL-level idempotency.

The poller asks each adapter for candidate documents and filters out URLs the
catalog has already processed (cheap pre-download dedup). Content-level
idempotency (SHA-256) is enforced later, after download, in the pipeline.
"""

from __future__ import annotations

import hashlib

from .. import db
from .adapters.base import DocumentRef, RIAdapter


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def discover_new(adapters: list[RIAdapter], db_path: str) -> list[DocumentRef]:
    db.init_db(db_path)
    new: list[DocumentRef] = []
    with db.connect(db_path) as conn:
        for adapter in adapters:
            for ref in adapter.discover():
                existing = db.find_by_url_hash(conn, url_hash(ref.pdf_url))
                if existing and existing["status"] in ("processed", "downloaded"):
                    continue
                new.append(ref)
    return new
