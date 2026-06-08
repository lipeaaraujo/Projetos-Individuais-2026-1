"""RI adapter interface.

An adapter knows how to discover candidate document URLs for one company's
Investor Relations results center. `discover()` scans the configured results
page for PDF links and merges in any curated known-document URLs. Idempotency
and download/extraction happen downstream in the pipeline, so adapters may
freely return already-seen documents.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

PDF_LINK_RE = re.compile(
    r'https?://[^\s"\'<>]*(?:mzfilemanager|\.pdf)[^\s"\'<>]*', re.I
)


@dataclass(frozen=True)
class DocumentRef:
    pdf_url: str
    source_page: str
    empresa: str | None = None
    title: str | None = None


class RIAdapter(ABC):
    name: str
    empresa: str
    results_page: str
    known_documents: tuple[DocumentRef, ...] = ()

    def __init__(self, *, user_agent: str, timeout: float):
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout

    def _scan_page(self) -> list[DocumentRef]:
        try:
            r = httpx.get(
                self.results_page,
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=True,
            )
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        urls = dict.fromkeys(
            urljoin(self.results_page, u) for u in PDF_LINK_RE.findall(r.text)
        )
        return [
            DocumentRef(pdf_url=u, source_page=self.results_page, empresa=self.empresa)
            for u in urls
        ]

    @abstractmethod
    def discover(self) -> list[DocumentRef]:
        ...


class MziqAdapter(RIAdapter):
    """Default discovery: static page scan + curated known documents.

    MZIQ-platform results centers load their full document list via an
    authenticated catalog API. The static scan captures whatever links the page
    exposes; `known_documents` carries the operational previews we already know
    about. Swapping in an authenticated catalog client or a headless renderer
    only requires overriding `discover()`.
    """

    def discover(self) -> list[DocumentRef]:
        refs = list(self.known_documents) + self._scan_page()
        seen, out = set(), []
        for ref in refs:
            if ref.pdf_url not in seen:
                seen.add(ref.pdf_url)
                out.append(ref)
        return out
