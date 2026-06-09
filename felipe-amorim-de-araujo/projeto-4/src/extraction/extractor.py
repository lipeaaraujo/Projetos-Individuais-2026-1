"""LLM extraction engine (Gemini, multimodal).

Sends the raw PDF to Gemini with the semantic contract as the output schema and
a system prompt that enforces the business rules. No regex/coordinate parsing —
resilience to layout changes comes from the model reading the document as-is.
"""

from __future__ import annotations

import json
import time

from ..contract import PreviaOperacional
from .chunking import ChunkPlan, plan_extraction

MAX_RETRIES = 3
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

SYSTEM_PROMPT = """\
Você é um motor de extração de dados para o Relatório de Conjuntura do Setor \
Habitacional do Ministério das Cidades. Recebe a Prévia Operacional (ou \
release de resultados) de uma incorporadora em PDF e extrai dados operacionais \
em JSON estrito, seguindo o schema fornecido.

REGRAS INVIOLÁVEIS:
1. VALORES ABSOLUTOS APENAS. Os documentos de RI destacam variações percentuais \
de marketing (ex.: "VGV +13,9%", "vendas -12% vs 2T25"). IGNORE essas variações. \
Extraia somente os valores brutos absolutos (VGV em R$, número de unidades). \
A única porcentagem aceita é o %Co (participação societária da empresa no projeto).
2. AUSENTE = null. Se uma métrica não estiver explicitamente no documento, \
retorne null. NUNCA invente, estime ou use 0 no lugar de um valor ausente.
3. UNIDADES: valores monetários (VGV) em R$ MIL (milhares de reais). Se o \
documento reportar em R$ milhões, converta para mil (multiplique por 1000). \
Contagem de unidades é número inteiro.
4. LANÇAMENTOS vs VENDAS são blocos distintos. Vendas = vendas líquidas \
contratadas no trimestre. Não misture.
5. empresa/ano/trimestre são obrigatórios. Determine o trimestre e ano de \
referência do documento (ex.: "Prévia 1T26" -> ano=2026, trimestre=1).
6. Responda SOMENTE com o JSON do schema, sem texto adicional, sem markdown.
"""


def _build_prompt(plan: ChunkPlan) -> str:
    schema = json.dumps(PreviaOperacional.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        f"{SYSTEM_PROMPT}\n\nJSON Schema do contrato semântico:\n{schema}\n\n"
        f"(estratégia de segmentação aplicada: {plan.strategy} — {plan.reason})\n"
        "Extraia os dados do PDF anexo e responda com o JSON correspondente."
    )


def extract(pdf_bytes: bytes, *, api_key: str, model: str) -> tuple[PreviaOperacional, ChunkPlan]:
    """Extract a validated PreviaOperacional from PDF bytes. Raises on failure."""
    from google import genai
    from google.genai import types

    plan = plan_extraction(pdf_bytes)
    if plan.page_count == 0:
        raise ValueError("PDF inválido ou vazio (0 páginas)")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json", temperature=0.0
    )
    contents = [
        types.Part.from_bytes(data=plan.pdf_bytes, mime_type="application/pdf"),
        _build_prompt(plan),
    ]

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            raw = (response.text or "").strip()
            if not raw:
                raise ValueError("Resposta vazia do modelo")
            return PreviaOperacional.model_validate_json(raw), plan
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES and any(t in str(e) for t in TRANSIENT):
                time.sleep(2 ** attempt)  # backoff exponencial: 2s, 4s
                continue
            raise
    raise last_err  # unreachable
