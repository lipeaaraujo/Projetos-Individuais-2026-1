from .base import DocumentRef, MziqAdapter


class TrisulAdapter(MziqAdapter):
    name = "trisul"
    empresa = "Trisul"
    results_page = "https://ri.trisul-sa.com.br/informacoes-financeiras/central-de-resultados/"
    known_documents = (
        DocumentRef(
            pdf_url="https://api.mziq.com/mzfilemanager/v2/d/b39881f3-1c97-405a-9650-9af4218e46bd/cd21a399-e28d-6ea7-b2e4-f2e08ab635f4?origin=2",
            source_page=results_page,
            empresa="Trisul",
            title="Prévia Operacional 1T26",
        ),
    )
