# Pipeline UDA — Conjuntura do Setor Habitacional

Pipeline de Engenharia e Análise de Dados Não Estruturados (UDA) que coleta
*Prévias Operacionais* em PDF das Centrais de Resultados (RI) de incorporadoras,
extrai os **valores operacionais absolutos** via LLM (Gemini, multimodal) sob um
**contrato semântico** e os serve por uma **API REST** para alimentar o
Relatório de Conjuntura do Ministério das Cidades.

O diferencial é a **resiliência a layout**: o mesmo extrator funciona em PDFs de
empresas e formatos diferentes porque a IA lê o documento como ele é — sem regex,
sem coordenadas de pixel.

## Arquitetura (3 camadas obrigatórias)

```
RI (Central de Resultados)
        │  polling diário + idempotência por hash
        ▼
┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│  A. Ingestão        │   │  B. Processamento     │   │  C. Serviço         │
│  adapters + poller  │──▶│  Gemini + contrato    │──▶│  API REST (FastAPI) │
│  + scheduler        │   │  semântico (Pydantic) │   │  conjuntura/série   │
└─────────────────────┘   └──────────────────────┘   └─────────────────────┘
        │                          │                          │
        └──────────── Catálogo de Dados + Linhagem (SQLite) ──┘
```

| Camada | Arquivo | Responsabilidade |
|---|---|---|
| Contrato semântico | `src/contract.py` | Schema Pydantic que blinda o banco: valores absolutos, ausente→NULL, tipos corretos |
| Catálogo + linhagem | `src/db.py` | `documents` (idempotência por SHA-256) + `metrics` (FK→documento = linhagem) |
| Ingestão | `src/ingestion/` | adapters de RI, poller idempotente, scheduler diário |
| Extração (UDA) | `src/extraction/` | decisão de *chunking* + chamada multimodal ao Gemini |
| Orquestração | `src/pipeline.py` | discover → download → hash → extract → persist |
| Serviço | `src/api/main.py` | endpoints REST com filtros e linhagem |

## Setup

```bash
cd felipe-amorim-de-araujo/projeto-4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # e preencha GEMINI_API_KEY
```

Obtenha a chave em https://aistudio.google.com/apikey (também aceita `GOOGLE_API_KEY`).

## Uso

```bash
# 1. Roda um ciclo de polling (descobre PDFs novos nas RIs e processa)
python scripts/run_poll_once.py

# 2. Ingestão pontual de um PDF (URL ou caminho local)
python scripts/ingest_pdf.py "https://api.mziq.com/.../previa.pdf" --empresa Direcional
python scripts/ingest_pdf.py data/pdfs/direcional_previa_1t26.pdf

# 3. Observação contínua (scheduler diário, roda um ciclo na partida)
python -m src.ingestion.scheduler

# 4. Sobe a API
uvicorn src.api.main:app --reload
```

## API

| Endpoint | Descrição |
|---|---|
| `GET /api/conjuntura?empresa=MRV&ano=2025&trimestre=3&tipo=lancamentos` | Métricas absolutas filtradas, com linhagem |
| `GET /api/series?empresa=Direcional&tipo=lancamentos` | Série temporal com variações **QoQ/YoY calculadas pelo banco** a partir dos absolutos |
| `GET /api/documents?empresa=MRV` | Catálogo de documentos coletados + status/linhagem |
| `GET /health` | Healthcheck |

Exemplo:

```bash
curl "http://localhost:8000/api/conjuntura?empresa=Direcional&ano=2026&trimestre=1"
```

```json
{
  "count": 2,
  "resultados": [{
    "empresa": "Direcional", "ano": 2026, "trimestre": 1, "tipo": "lancamentos",
    "vgv_total_mil": 6400000.0, "unidades": 12000, "percentual_co": null,
    "lineage": {
      "document_id": 1,
      "pdf_url": "https://api.mziq.com/.../previa.pdf",
      "source_page": "https://ri.direcional.com.br/.../central-de-resultados/"
    }
  }]
}
```

## Contrato semântico (o coração)

`PreviaOperacional` (em `src/contract.py`) é injetado no prompt como JSON Schema
e valida a resposta antes de qualquer escrita no banco. Regras blindadas:

- **Valores absolutos apenas.** As Prévias destacam variações de marketing
  (`VGV +13,9%`); o contrato **não as modela**. Guardamos só os brutos; o banco
  calcula o histórico real (`/api/series`).
- **Ausente → NULL.** Todo campo de negócio é `Optional`; o modelo retorna `null`
  (nunca 0, nunca chute).
- **Unidades estáveis.** VGV em R$ mil; unidades em inteiro.

## Resiliência (≥2 layouts)

O pipeline foi validado em PDFs reais de formatos distintos — ex.: Prévia
Operacional da **Direcional** (relatório de 7 páginas) e da **Trisul**
(4 páginas), além do Boletim em tabela e do Release de Resultados da Cury
(37 páginas). O mesmo extrator multimodal processa todos sem regras de layout.

## Testes

```bash
pytest -q
```

Cobrem: validação do contrato, persistência + linhagem, idempotência do pipeline
(o mesmo PDF não chama o LLM duas vezes) e os endpoints da API.
