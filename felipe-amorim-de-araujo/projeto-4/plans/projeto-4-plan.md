# Plano de Implementação — Pipeline UDA do Setor Habitacional (Projeto 4)

Registro completo do plano que guiou a construção: contexto, decisões,
arquitetura detalhada, descobertas de campo, bugs corrigidos e validação.

---

## 1. Contexto e objetivo

Desafio de **Análise de Dados Não Estruturados (UDA)**: o Ministério das Cidades
produz periodicamente um *Relatório de Conjuntura do Setor Habitacional* que
depende de dados operacionais das principais incorporadoras — dados que ficam
pulverizados em *Prévias Operacionais* (PDF) publicadas trimestralmente nos
portais de Relações com Investidores (RI).

Construir um pipeline que:
1. **Observe continuamente** as Centrais de Resultados (RI) e detecte PDFs novos.
2. **Extraia semanticamente** (via LLM, sem regras rígidas de layout) os
   **valores absolutos** — ignorando os percentuais de marketing.
3. **Sirva** os dados por uma **API REST** estruturada.

Diferencial avaliado: **resiliência a variações de layout** — o mesmo script tem
de rodar em ≥2 layouts de empresas diferentes.

Prazo: entrega 08.06 (mesmo dia do desenvolvimento) → viés para entregável
funcional e bem arquitetado em vez de amplitude.

---

## 2. Decisões iniciais (e justificativa)

| Decisão | Escolha | Justificativa |
|---|---|---|
| LLM | **Gemini** (`google-genai`, `gemini-2.5-flash`) | Multimodal nativo — lê o PDF bruto como imagem/texto, fonte da resiliência a layout; tier gratuito; já presente no stack do projeto-1 |
| Fonte de dados | **Baixar Prévias reais** das RIs | Autenticidade; prova real de resiliência |
| Gatilho de ingestão | **Polling diário (APScheduler)** + 2 adapters reais | RIs não expõem webhook/RSS padronizado; prévias são trimestrais → varredura diária basta e não sobrecarrega |
| Chunking | **Full-scan**, fallback semântico p/ docs longos | Prévias são curtas (≤~20p); full-scan elimina o parsing frágil onde nascem os erros de layout |
| Extração | **Motor nativo** (PyMuPDF + Gemini) em vez de LOTUS/DocETL | Controle total sobre contrato e prompt (foco da nota); dependências mínimas |
| Storage | **SQLite** | Dataset pequeno, relacional (joins de linhagem), zero-ops, o arquivo `.db` é o catálogo |
| API | **FastAPI + uvicorn** | Já no stack; validação de query + docs automáticas |
| Stack | Python 3.11+ (rodou em 3.14), Pydantic v2, httpx, BeautifulSoup, PyMuPDF | Alinhado ao projeto-1 |

---

## 3. Arquitetura

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

### Estrutura de arquivos

```
projeto-4/
├── src/
│   ├── config.py                  # Settings (pydantic-settings), GEMINI_API_KEY/GOOGLE_API_KEY
│   ├── contract.py                # CONTRATO SEMÂNTICO (Pydantic) — núcleo avaliado
│   ├── db.py                      # Catálogo (documents) + métricas (metrics) + linhagem
│   ├── pipeline.py                # Orquestração discover→download→hash→extract→persist
│   ├── ingestion/
│   │   ├── adapters/
│   │   │   ├── base.py            # RIAdapter (ABC) + MziqAdapter + DocumentRef
│   │   │   ├── direcional.py      # adapter real (layout relatório)
│   │   │   └── trisul.py          # adapter real (layout diferente)
│   │   ├── poller.py              # descoberta + dedup por URL
│   │   └── scheduler.py           # APScheduler diário (roda 1x na partida)
│   ├── extraction/
│   │   ├── chunking.py            # full-scan vs semantic-chunk
│   │   └── extractor.py           # chamada multimodal Gemini + retry/backoff
│   └── api/main.py                # endpoints REST
├── scripts/
│   ├── run_poll_once.py           # 1 ciclo de polling
│   └── ingest_pdf.py              # ingestão pontual (URL ou arquivo local)
├── tests/                         # 18 testes (contrato, db/linhagem, idempotência, API)
├── data/{pdfs/, catalog.db}       # gitignored
├── README.md · requirements.txt · .env.example · .gitignore
```

---

## 4. Contrato Semântico (`src/contract.py`) — o núcleo

`PreviaOperacional` (Pydantic v2). Três papéis: especifica (vira JSON Schema no
prompt), valida (rejeita antes de persistir), documenta as regras.

Modelos:
- `Trimestre(IntEnum)`: 1–4.
- `MetricaOperacional`: `vgv_total_mil`, `vgv_co_mil` (float, R$ mil),
  `unidades` (int), `percentual_co` (float 0–100). Todos `Optional` → NULL.
  Validator rejeita VGV negativo.
- `PreviaOperacional`: `empresa` (str, normalizada), `ano` (int 2000–2100),
  `trimestre` (Trimestre) — **obrigatórios**; `lancamentos` e `vendas`
  (MetricaOperacional); `observacoes` (Optional). Validator normaliza empresa.

Regras blindadas:

| Regra | Motivo |
|---|---|
| Valores absolutos apenas | Ignorar % de marketing; banco calcula histórico. As variações **não existem** no schema |
| Ausente → NULL | Todo campo de negócio Optional; prompt proíbe 0/chute |
| %Co é a única % | Participação societária = fato estrutural, não variação temporal |
| VGV em R$ mil; unidades int | Unidade estável evita ambiguidade milhões/mil entre layouts |
| empresa/ano/trimestre obrigatórios | Sem eixo temporal a linha é inútil → falhar, não chutar |

---

## 5. Catálogo + Linhagem (`src/db.py`)

SQLite, `PRAGMA foreign_keys=ON`, `row_factory=Row`.

- **`documents`**: `id`, `empresa`, `pdf_url`, `source_page` (página de RI onde o
  link foi achado), `sha256` (UNIQUE — chave de idempotência), `url_hash`,
  `file_path`, `title`, `status` (discovered|downloaded|processed|failed|skipped),
  `error`, `detected_at`, `processed_at`.
- **`metrics`**: `id`, `document_id` (FK→documents, ON DELETE CASCADE),
  `empresa`, `ano`, `trimestre`, `tipo` (lancamentos|vendas),
  `vgv_total_mil`, `vgv_co_mil`, `unidades`, `percentual_co`, `observacoes`,
  `extracted_at`. **`UNIQUE(empresa, ano, trimestre, tipo)`** → upsert idempotente.
  Índice em `(empresa, ano, trimestre)`.

Helpers: `init_db`, `connect` (context manager), `find_by_sha256`,
`find_by_url_hash`, `register_document`, `mark_downloaded/processed/failed/skipped`,
`delete_document`, `persist_extraction` (grava os 2 blocos via upsert).

**Linhagem** = junção `metrics.document_id → documents.id`: qualquer número da API
rastreia até o PDF de origem e a página de RI.

---

## 6. Ingestão (`src/ingestion/`)

- `DocumentRef`: `pdf_url`, `source_page`, `empresa?`, `title?`.
- `RIAdapter` (ABC): `name`, `empresa`, `results_page`, `known_documents`,
  `_scan_page()` (regex `mzfilemanager|.pdf` na página estática), `discover()`.
- `MziqAdapter`: `discover()` = `known_documents` + scan da página, dedup por URL.
- Adapters reais: `DirecionalAdapter`, `TrisulAdapter`.
- `poller.discover_new()`: para cada ref calcula `url_hash` (sha256 da URL) e
  descarta o que já está `processed/downloaded` (dedup pré-download).
- `scheduler.start()`: APScheduler cron diário (`POLL_HOUR`), timezone
  America/Sao_Paulo, roda 1 ciclo na partida.

---

## 7. Extração (`src/extraction/`)

- `chunking.plan_extraction(pdf_bytes)`:
  - ≤ 20 páginas → **full-scan** (envia PDF inteiro).
  - > 20 páginas → **semantic-chunk**: retém só páginas cujo texto contém
    keywords (vgv, lançamento, vendas, unidades, %co, prévia operacional…);
    sem páginas relevantes → fallback full-scan.
  - Retorna `ChunkPlan(strategy, pdf_bytes, page_count, kept_pages, reason)`.
- `extractor.extract(pdf_bytes, api_key, model)`:
  - guarda 0 páginas → erro claro.
  - `genai.Client(...).models.generate_content(model, contents=[Part.from_bytes(
    pdf, "application/pdf"), prompt], config={response_mime_type:"application/json",
    temperature:0.0})`.
  - prompt de sistema reforça: valores absolutos, ausente→null, R$ mil,
    lançamentos≠vendas, empresa/ano/trimestre obrigatórios, responder só JSON.
    JSON Schema do contrato injetado no prompt.
  - valida com `PreviaOperacional.model_validate_json`.
  - **retry com backoff exponencial** (2s, 4s; até 3 tentativas) p/ erros
    transitórios: 503/UNAVAILABLE/429/RESOURCE_EXHAUSTED/500/INTERNAL.

---

## 8. Orquestração (`src/pipeline.py`)

`ingest_ref(ref, settings)`:
1. `register_document` (status discovered).
2. `_download` (suporta `file://` p/ ingestão local; httpx p/ URLs).
3. `sha256` do conteúdo → `find_by_sha256`:
   - existe e `processed` → `mark_skipped`, retorna **skipped** (sem chamar LLM).
   - existe mas não concluído → `delete_document` da linha nova, **reaproveita**
     a linha existente (retry; evita choque do UNIQUE em sha256).
4. grava PDF, `mark_downloaded`.
5. sem `GEMINI_API_KEY` → `mark_failed`.
6. `extract(...)` → on erro `mark_failed`.
7. `persist_extraction` + `mark_processed`.

`run_pipeline(adapters, settings)`: `discover_new` → `ingest_ref` em cada novo.

Idempotência em **duas barreiras antes do LLM**: por URL (poller) e por conteúdo
SHA-256 (pipeline).

---

## 9. API (`src/api/main.py`) — FastAPI, lifespan handler

| Endpoint | Função |
|---|---|
| `GET /api/conjuntura?empresa&ano&trimestre&tipo` | métricas absolutas filtradas + `lineage` |
| `GET /api/series?empresa&tipo` | série temporal com **QoQ/YoY calculados dos absolutos** (ciente de lacunas → null) |
| `GET /api/documents?empresa` | catálogo + status + linhagem |
| `GET /health` | healthcheck |

`/api/series` é a prova do requisito "extrair absolutos p/ o banco calcular o
histórico real" — `_pct(curr, prev)` calcula a variação, não repassa o % do
marketing.

---

## 10. Testes (`tests/`) — 18, todos passando

- `test_contract.py`: mínimos válidos, absolutos preservados, VGV negativo
  rejeitado, %Co fora de 0–100 rejeitado, trimestre inválido, normalização de
  empresa, schema exportável.
- `test_db.py`: persistência + linhagem (join), upsert idempotente (1 linha por
  trimestre/tipo), `find_by_sha256`.
- `test_pipeline.py`: 1ª ingestão processa; 2ª ingestão do mesmo conteúdo →
  skipped e **LLM não é chamado de novo** (extract mockado, conta chamadas).
- `test_api.py`: health, filtros de conjuntura, isolamento por empresa, série
  com YoY calculado, documentos com linhagem.

---

## 11. Descobertas de campo (durante a implementação)

- **MZIQ SPAs**: as Centrais de Resultados (MRV, Cury, Direcional, Trisul…) são
  SPAs WordPress da plataforma MZIQ. A lista completa de documentos carrega via
  API de catálogo `apicatalog.mziq.com/filemanager` que exige autenticação
  (retorna **401**). Variáveis no HTML: `fmId` (UUID da empresa), `fmName`,
  `fmBase`; categorias incluem `central_de_resultados_previa`. Os PDFs ficam em
  `api.mziq.com/mzfilemanager/v2/d/<uuid>/<uuid>?origin=2`.
- Estratégia adotada: scan estático captura links expostos + `known_documents`
  com as Prévias reais; trocar por cliente autenticado/headless = sobrescrever
  `discover()`. As páginas de Direcional/Trisul expõem links extras estaticamente
  (descoberta dinâmica real funciona).
- **Prévias reais usadas** (2 layouts distintos):
  - Direcional 1T26 — `ada9bc2c-…/b9e3e792-…` (relatório, 7p, tabelas c/ colunas
    de variação).
  - Trisul 1T26 — `b39881f3-…/cd21a399-…` (4p, estrutura diferente).
  - Extras p/ resiliência: Cury Release 1T26 (37p → semantic-chunk), Boletim (1p).

---

## 12. Bugs encontrados e corrigidos

| Bug | Sintoma | Correção |
|---|---|---|
| `config.py` com hack `Field_default` | módulo confuso/instável | reescrito limpo com `Field(alias=...)` |
| `enabled_adapters` default `"mrv,cury"` | `Adapters: []` no poll (registry só tem direcional/trisul) | default → `"direcional,trisul"` (+ `.env.example`) |
| `sha256` UNIQUE no retry | reingestão de conteúdo com linha `failed` anterior estourava `IntegrityError` | dedup trata processed=skip vs não-concluído=retry (delete linha nova, reaproveita existente) |
| 503 transitório do Gemini | docs falhavam por sobrecarga do modelo | retry com backoff exponencial p/ 503/429/500 |
| link não-PDF no scan | Gemini retornava 400 "no pages" | guarda 0 páginas no extractor |
| FastAPI `on_event` deprecado | warning | migrado p/ `lifespan` |

---

## 13. Validação final (execução real com Gemini)

- **Resiliência**: mesmo extrator processou Direcional + Trisul (layouts
  distintos) → ambos `processed`.
- **Valores absolutos verificados** contra o texto do PDF da Direcional:

  | Métrica 1T26 | Extraído (R$ mil/un.) | No PDF |
  |---|---|---|
  | Lançamentos VGV 100% | 1.005.800 | "VGV Lançado (VGV 100%) 1.005,8" (milhões) |
  | Lançamentos VGV %Co | 862.400 | "(R$ 862 milhões % Companhia)" |
  | Lançamentos unidades | 3.109 | "Unidades Lançadas 3.109" |
  | Vendas VGV 100% | 1.582.000 | "VGV Líquido Contratado 1.582,0" |
  | Vendas VGV %Co | 1.352.000 | "(% Companhia) 1.352,0" |
  | Vendas unidades | 4.848 | "Unidades Contratadas 4.848" |

  O modelo ignorou as colunas de variação (`-47,1%`, `-35,0%`) ao lado dos
  absolutos, converteu milhões→mil, e separou lançamentos×vendas.
- **Idempotência**: reingestão → `skipped: conteúdo já processado`, sem LLM.
- **18/18 testes** passando.

---

## 14. Limitações / evolução

- Descoberta dinâmica limitada aos links estáticos; cobertura total das Centrais
  MZIQ pede cliente de catálogo autenticado ou Playwright sob a mesma interface
  `RIAdapter`.
- SQLite serve ao escopo; volume maior → Postgres com o mesmo schema.
