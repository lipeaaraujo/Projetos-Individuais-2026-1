"""The semantic contract is the quality gate; these tests pin its rules."""

import pytest
from pydantic import ValidationError

from src.contract import PreviaOperacional, Trimestre


def test_minimal_valid_extraction():
    p = PreviaOperacional.model_validate(
        {"empresa": "MRV", "ano": 2026, "trimestre": 1}
    )
    assert p.empresa == "MRV"
    assert p.trimestre == Trimestre.PRIMEIRO
    # missing metrics -> None (persisted as NULL), never 0/guessed
    assert p.lancamentos.vgv_total_mil is None
    assert p.vendas.unidades is None


def test_absolute_values_preserved():
    p = PreviaOperacional.model_validate(
        {
            "empresa": "Direcional",
            "ano": 2026,
            "trimestre": 1,
            "lancamentos": {"vgv_total_mil": 6400000, "unidades": 12000},
            "vendas": {"vgv_co_mil": 5400000, "percentual_co": 84.0},
        }
    )
    assert p.lancamentos.vgv_total_mil == 6400000
    assert p.vendas.percentual_co == 84.0


def test_negative_vgv_rejected():
    with pytest.raises(ValidationError):
        PreviaOperacional.model_validate(
            {"empresa": "X", "ano": 2025, "trimestre": 2,
             "lancamentos": {"vgv_total_mil": -100}}
        )


def test_percentual_co_bounds():
    with pytest.raises(ValidationError):
        PreviaOperacional.model_validate(
            {"empresa": "X", "ano": 2025, "trimestre": 2,
             "vendas": {"percentual_co": 150}}
        )


def test_invalid_quarter_rejected():
    with pytest.raises(ValidationError):
        PreviaOperacional.model_validate({"empresa": "X", "ano": 2025, "trimestre": 5})


def test_empresa_normalized():
    p = PreviaOperacional.model_validate(
        {"empresa": "  Plano   &  Plano ", "ano": 2025, "trimestre": 3}
    )
    assert p.empresa == "Plano & Plano"


def test_empty_empresa_rejected():
    with pytest.raises(ValidationError):
        PreviaOperacional.model_validate({"empresa": "   ", "ano": 2025, "trimestre": 1})


def test_schema_is_exportable():
    # the schema is injected into the prompt; it must serialize
    schema = PreviaOperacional.model_json_schema()
    assert "empresa" in schema["properties"]
    assert "lancamentos" in schema["properties"]
