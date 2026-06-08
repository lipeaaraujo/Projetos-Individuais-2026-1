"""Document segmentation strategy.

Decision: **full-scan by default**. Prévias Operacionais are short (typically
≤ ~20 pages) and Gemini reads the raw PDF multimodally, so layout-agnostic
whole-document extraction is both simplest and most resilient. For the rare
long document we fall back to semantic chunking: keep only pages whose text
mentions operational/financial keywords, dropping boilerplate before the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF

FULL_SCAN_PAGE_LIMIT = 20

KEYWORDS = (
    "vgv", "valor geral de vendas", "lançamento", "lancamento", "vendas",
    "unidades", "%co", "prévia operacional", "previa operacional",
    "vendas líquidas", "vendas liquidas", "vendas contratadas",
)


@dataclass
class ChunkPlan:
    strategy: str            # "full-scan" | "semantic-chunk"
    pdf_bytes: bytes         # the bytes to send to the LLM (whole or filtered)
    page_count: int
    kept_pages: int
    reason: str


def plan_extraction(pdf_bytes: bytes) -> ChunkPlan:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count

    if page_count <= FULL_SCAN_PAGE_LIMIT:
        doc.close()
        return ChunkPlan(
            strategy="full-scan",
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            kept_pages=page_count,
            reason=f"{page_count} páginas ≤ limite {FULL_SCAN_PAGE_LIMIT}: envio integral.",
        )

    relevant = [
        i for i in range(page_count)
        if any(k in doc.load_page(i).get_text().lower() for k in KEYWORDS)
    ]
    if not relevant:
        doc.close()
        return ChunkPlan(
            strategy="full-scan",
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            kept_pages=page_count,
            reason="Nenhuma página relevante detectada por keywords: fallback full-scan.",
        )

    out = fitz.open()
    out.insert_pdf(doc, from_page=relevant[0], to_page=relevant[0])
    for i in relevant[1:]:
        out.insert_pdf(doc, from_page=i, to_page=i)
    filtered = out.tobytes()
    out.close()
    doc.close()
    return ChunkPlan(
        strategy="semantic-chunk",
        pdf_bytes=filtered,
        page_count=page_count,
        kept_pages=len(relevant),
        reason=f"Documento longo ({page_count}p): retidas {len(relevant)} páginas com métricas.",
    )
