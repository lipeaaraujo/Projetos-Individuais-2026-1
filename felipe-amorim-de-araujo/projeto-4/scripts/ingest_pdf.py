"""Ingest a single PDF by URL or local path (for testing / ad-hoc ingestion).

Usage:
    python scripts/ingest_pdf.py <url-or-path> [--empresa NOME]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.ingestion.adapters.base import DocumentRef
from src.pipeline import ingest_ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="URL http(s) ou caminho local do PDF")
    ap.add_argument("--empresa", default=None, help="Dica de nome da empresa (opcional)")
    args = ap.parse_args()

    settings = get_settings()
    src = args.source

    if src.startswith("http"):
        ref = DocumentRef(pdf_url=src, source_page=src, empresa=args.empresa, title=src)
    else:
        path = Path(src).resolve()
        ref = DocumentRef(
            pdf_url=path.as_uri(), source_page="local", empresa=args.empresa, title=path.name
        )

    res = ingest_ref(ref, settings)
    print(f"{res.status}: {res.detail}")


if __name__ == "__main__":
    main()
