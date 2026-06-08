"""Semantic Contract: the Pydantic schema the LLM must satisfy when extracting
an operational preview. It is injected into the prompt as JSON Schema, validates
the response before persistence, and encodes the business rules:

- Absolute values only. IR decks highlight % deltas ("VGV +13.9%") — those are
  marketing and are NOT modelled. The DB computes deltas itself. The only
  percentage stored is `percentual_co` (the company's ownership stake, %Co),
  which is a structural fact of the deal, not a temporal delta.
- Missing -> NULL. Every business field is Optional; the model returns null
  (not 0, not a guess) when a figure is absent.
- Money in BRL thousands (R$ mil); units are integers.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Trimestre(IntEnum):
    PRIMEIRO = 1
    SEGUNDO = 2
    TERCEIRO = 3
    QUARTO = 4


class MetricaOperacional(BaseModel):
    """One operational block (launches or sales) for a quarter. Money in R$ mil."""

    vgv_total_mil: Optional[float] = Field(
        default=None,
        description="VGV TOTAL (100% do projeto) em R$ mil. Valor absoluto bruto, nunca variação percentual.",
    )
    vgv_co_mil: Optional[float] = Field(
        default=None,
        description="VGV %Co (parcela proporcional à participação da empresa) em R$ mil. Valor absoluto.",
    )
    unidades: Optional[int] = Field(
        default=None,
        description="Número absoluto de unidades (lançadas ou vendidas). Inteiro, nunca percentual.",
    )
    percentual_co: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="%Co — participação societária média da empresa no(s) projeto(s), em pontos percentuais (0–100).",
    )

    @field_validator("vgv_total_mil", "vgv_co_mil", mode="before")
    @classmethod
    def _reject_negative_money(cls, v):
        if v is not None and v < 0:
            raise ValueError("Valores de VGV não podem ser negativos")
        return v


class PreviaOperacional(BaseModel):
    """Top-level extraction object: one company's preview for one quarter.

    empresa/ano/trimestre are mandatory — without them the row cannot be placed
    on the time axis, so the model must fail rather than guess them.
    """

    empresa: str = Field(
        description="Nome curto e canônico da incorporadora (ex.: 'MRV', 'Cury', 'Tenda', 'Plano & Plano', 'Direcional', 'Pacaembu'). Nome comercial, não razão social.",
    )
    ano: int = Field(ge=2000, le=2100, description="Ano de referência do trimestre.")
    trimestre: Trimestre = Field(description="Trimestre de referência (1, 2, 3 ou 4).")

    lancamentos: MetricaOperacional = Field(
        default_factory=MetricaOperacional,
        description="Métricas absolutas de LANÇAMENTOS no trimestre.",
    )
    vendas: MetricaOperacional = Field(
        default_factory=MetricaOperacional,
        description="Métricas absolutas de VENDAS (líquidas contratadas) no trimestre.",
    )
    observacoes: Optional[str] = Field(
        default=None,
        description="Notas livres úteis para auditoria (ex.: 'documento reporta apenas operação Brasil'). Opcional.",
    )

    @field_validator("empresa")
    @classmethod
    def _normalize_empresa(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("empresa não pode ser vazia")
        return v
