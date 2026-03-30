import re
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 8


@dataclass
class Offer:
    store: str
    title: str
    price: str
    url: str


def verify_price(book_title: str) -> list[Offer]:
    offers = []
    checkers = [
        _search_estante_virtual,
        _search_americanas,
    ]
    for checker in checkers:
        try:
            result = checker(book_title)
            if result:
                offers.append(result)
        except Exception:
            pass
    return offers


def _search_estante_virtual(title: str) -> Offer | None:
    url = f"https://www.estantevirtual.com.br/busca?q={quote_plus(title)}"
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    soup = BeautifulSoup(resp.text, "html.parser")

    price_tag = soup.select_one("[class*='price'], [class*='preco'], [itemprop='price']")
    if not price_tag:
        return None

    price = _parse_price(price_tag.get_text())
    if not price:
        return None

    title_tag = soup.select_one("[class*='title'], [class*='titulo'], h2, h3")
    found_title = title_tag.get_text(strip=True) if title_tag else title

    return Offer(store="Estante Virtual", title=found_title, price=price, url=url)


def _search_americanas(title: str) -> Offer | None:
    url = f"https://www.americanas.com.br/busca/{quote_plus(title)}?category=livros"
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    soup = BeautifulSoup(resp.text, "html.parser")

    price_tag = soup.select_one("[class*='price'], [class*='Price']")
    if not price_tag:
        return None

    price = _parse_price(price_tag.get_text())
    if not price:
        return None

    return Offer(store="Americanas", title=title, price=price, url=url)


def _parse_price(text: str) -> float | None:
    text = text.replace("R$", "").strip()
    match = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))", text)
    if not match:
        return None
    value = match.group(1).replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None
