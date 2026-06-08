from .base import DocumentRef, MziqAdapter


class DirecionalAdapter(MziqAdapter):
    name = "direcional"
    empresa = "Direcional"
    results_page = "https://ri.direcional.com.br/informacoes-financeiras/central-de-resultados/"
    known_documents = (
        DocumentRef(
            pdf_url="https://api.mziq.com/mzfilemanager/v2/d/ada9bc2c-f7d0-4359-9eaf-851b679ab788/b9e3e792-da8b-5e49-f50f-4c097cf08623?origin=2",
            source_page=results_page,
            empresa="Direcional",
            title="Prévia Operacional 1T26",
        ),
    )
