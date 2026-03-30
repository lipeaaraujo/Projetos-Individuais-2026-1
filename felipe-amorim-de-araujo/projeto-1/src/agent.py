import os
import requests
from catalog_builder import RAGCatalog
from book_fetcher import search_book_metadata, Book
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DB_PATH = "data/chroma_db"
WORKS_URL = "https://openlibrary.org{work_key}.json"


class Agent:
    def __init__(self, db_path: str = DB_PATH):
        self._catalog = RAGCatalog(db_path=db_path)

    def recommend(self, read_books: list[str], k: int = 5) -> list[Book]:
        read_context: list[Book] = []
        for title in read_books:
            try:
                meta = search_book_metadata(title)
                read_context.append(meta)
            except Exception:
                book = Book(
                    title=title,
                    authors=[],
                    categories=[],
                    description="",
                    isbn="",
                    work_key=""
                )
                read_context.append(book)

        query = _build_rag_query(read_context)

        candidates = self._catalog.search_similar(query, k=k * 3, titles_to_remove=read_books)

        print(candidates)


def _build_rag_query(read_context: list[Book]) -> str:
    categories = set()
    descriptions = []
    for book in read_context:
        categories.update(book.categories)
        if book.description:
            descriptions.append(book.description[:200])
    return " ".join(list(categories)) + " " + " ".join(descriptions[:3])


def _enrich_candidates(candidates: list[Book]) -> list[Book]:
    enriched = []
    for book in candidates:
        work_key = book.get("work_key", "")
        if not work_key:
            enriched.append(book)
            continue
        try:
            resp = requests.get(WORKS_URL.format(work_key=work_key), timeout=8)
            data = resp.json()

            description = data.get("description", "")
            if isinstance(description, dict):
                description = description.get("value", "")

            categories = data.get("subjects", [])

            enriched.append({
                **book,
                "description": description,
                "categories": ", ".join(categories[:8]) if categories else book["categories"]
            })
        except Exception:
            enriched.append(book)

    return enriched
