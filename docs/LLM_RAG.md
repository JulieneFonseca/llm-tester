# Documento de Especificação — LLM RAG

**Projeto:** llm-tester
**Componente:** LLM com RAG (Retrieval-Augmented Generation)
**Arquivos-fonte:** `src/llm_tester/pipeline.py`, `src/llm_tester/config.py`
**Versão do documento:** 1.1
**Data:** 31/08/2026

---

## 1. Visão geral

A **LLM com RAG** é a abordagem em que a pergunta é respondida **com apoio de
contexto recuperado** de uma base jurídica local, em contraste com a **LLM
Padrão**, que responde diretamente, sem contexto suplementar.

O propósito no llm-tester é medir, de forma controlada, **o quanto o contexto
recuperado melhora a qualidade** (fidelidade, precisão vs. gabarito, completude e
redução de alucinação) e **qual o custo em latência** (overhead) que essa
recuperação introduz. Como as duas abordagens usam o **mesmo modelo**, a única
variável entre elas é a presença ou não do contexto.

---

## 2. Tipo de RAG utilizado

A arquitetura implementada é um **RAG "clássico" (naive/standard RAG)** de
recuperação única, com as seguintes características:

- **Recuperação densa (dense retrieval):** busca por similaridade de embeddings
  vetoriais (não há busca lexical/BM25 nem busca híbrida).
- **Passo único (single-shot):** um único ciclo de *retrieve → generate*, sem
  iteração, sem re-recuperação e sem agentes.
- **Diversificação por MMR:** por padrão, a seleção dos `top_k` trechos usa
  *Maximal Marginal Relevance* (relevância vs. diversidade), reduzindo trechos
  redundantes/duplicados. Configurável para similaridade pura.
- **Stuffing de contexto:** todos os trechos recuperados são concatenados e
  inseridos ("stuffed") diretamente no prompt.
- **Indexação persistente:** os embeddings são calculados uma vez e persistidos
  no vector store (ChromaDB), sendo reutilizados entre execuções.

Em resumo: **Dense Retrieval + MMR (Top-K diverso) + Context Stuffing + geração
única.**

---

## 3. Arquitetura RAG

### 3.1. Pipeline em duas fases

```
FASE 1 — INGESTÃO (offline, uma vez por base/tema)
  Documentos (PDF/TXT/CSV)
      │  PyPDFLoader / TextLoader / CSV loader interno
      │  (CSV: resumo automático de campos > limite_campo_chars)
      ▼
  Documentos carregados
      │  RecursiveCharacterTextSplitter (chunk_size / overlap)
      ▼
  Chunks
      │  HuggingFaceEmbeddings (all-MiniLM-L6-v2, local)
      ▼
  Vetores → ChromaDB (persist_directory, collection_name)

FASE 2 — CONSULTA (por pergunta, em tempo de execução)
  Pergunta
      │  embedding da pergunta + busca vetorial (Top-K)   ── cronometrado (retrieval)
      ▼
  Trechos recuperados (contexto)
      │  montagem do prompt RAG (contexto + pergunta)
      ▼
  LLM (Groq)                                              ── cronometrado (geração)
      ▼
  Resposta + fontes + tempos
```

### 3.2. Ingestão e indexação (`Pipeline.indexar_base`)

- **Carregamento:** varre o diretório da base recursivamente (`**/*.pdf`,
  `**/*.txt` e `**/*.csv`); PDFs via `PyPDFLoader`, textos via `TextLoader`
  (UTF-8) e CSVs via loader interno com resumo de campos grandes. Falhas em
  arquivos individuais são registradas e não abortam a ingestão.
- **CSV — resumo de campos grandes:** cada linha do CSV vira um `Document`. Se
  um campo excede `limite_campo_chars` (padrão **500**), ele é resumido
  automaticamente para até `resumo_max_chars` (padrão **500**) caracteres,
  preservando sentenças completas. Campos curtos (≤200 chars) são incluídos
  também nos metadados do documento para facilitar filtragem posterior.
  O **delimitador** é configurável (`delimitador`, padrão `;`, também aceita
  `,`, tab ou `"auto"`). Leitura em `utf-8-sig` (remove BOM). Configuração em
  `parametros.csv_base_juridica`.
- **Segmentação (chunking):** `RecursiveCharacterTextSplitter` com
  `chunk_size` (padrão **1500**) e `chunk_overlap` (padrão **300**), usando os
  separadores `["\n\n", "\n", ". ", " ", ""]` (preferência por quebras naturais).
- **Deduplicação e sanitização:** chunks sem conteúdo (texto `None`/vazio) são
  descartados, e chunks com conteúdo idêntico são removidos (mantém o primeiro),
  evitando redundância e falhas de validação.
- **Cronometragem:** cada fase (carga, chunking+dedup, indexação vetorial) é
  cronometrada e registrada no log e no relatório (`performance_indexacao`),
  com o total em segundos/minutos e o *throughput* em chunks/s.
- **Embeddings:** `HuggingFaceEmbeddings` com `all-MiniLM-L6-v2` (executado
  **localmente**, sem custo e sem chamada de rede).
- **Vector store:** ChromaDB, persistido em `persist_directory`
  (padrão `data/chroma_db`) sob a coleção `collection_name`
  (padrão `pensao_por_morte`).
- **Reuso vs. reindexação:** se já existe um vector store e `forcar=False`, a
  coleção é apenas carregada; com `forcar=True` (ou `--reindexar`), a base é
  reprocessada do zero.

### 3.3. Recuperação (`Pipeline._recuperar_trechos`)

- A busca é feita **direto na collection do ChromaDB** (`collection.query`),
  que retorna dicionários crus (`documents` + `metadatas` + `embeddings`).
  Isso evita que o LangChain instancie objetos `Document` a partir de
  registros com texto nulo (o que quebraria a validação do pydantic), e
  descarta trechos sem conteúdo automaticamente.
- `top_k` padrão = **8** — os oito trechos selecionados por pergunta.
- **MMR (padrão):** busca `fetch_k = top_k * 3` candidatos e aplica *Maximal
  Marginal Relevance* manualmente (com `numpy`), equilibrando relevância à
  pergunta e diversidade entre os selecionados via `mmr_lambda` (padrão
  **0.7**). Com `search_type="similarity"`, usa similaridade densa pura.

### 3.4. Geração (`Pipeline.executar_rag`)

1. **Retrieval:** via `_recuperar_trechos` (busca direta na collection + MMR);
   cronometrado em `tempo_retrieval_rag_s`.
2. **Montagem do contexto:** concatena os trechos separados por `\n\n---\n\n` e
   registra as fontes (trecho truncado em 500 caracteres + metadados).
3. **Prompt RAG:** preenche o template com `{dominio}`, `{contexto_recuperado}`
   e `{pergunta}`.
4. **Geração:** invoca a LLM (via `com_backoff`); cronometrada em
   `tempo_geracao_rag_s`.
5. **Retorno:** resposta, contexto, fontes e tempos (retrieval, geração e total).

### 3.5. Prompt RAG (padrão)

```
Você é um assistente jurídico especializado em {dominio}. Responda à pergunta
utilizando ESTRITAMENTE as informações fornecidas no contexto abaixo.

CONTEXTO:
{contexto_recuperado}

PERGUNTA:
{pergunta}
```

O prompt instrui o modelo a **restringir-se ao contexto recuperado**, o que é a
principal alavanca de fidelidade e redução de alucinação da abordagem RAG.

---

## 4. Configurações

Todas em `config.json` (seção `parametros`), com defaults em `config.py`:

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Modelo de embeddings (local) |
| `chunk_size` | `1500` | Tamanho do chunk na segmentação |
| `chunk_overlap` | `300` | Sobreposição entre chunks |
| `top_k` | `8` | Nº de trechos recuperados |
| `search_type` | `mmr` | Estratégia de busca (`mmr` ou `similarity`) |
| `mmr_lambda` | `0.7` | Peso relevância × diversidade no MMR (1.0 = só relevância) |
| `collection_name` | `pensao_por_morte` | Coleção no ChromaDB |
| `persist_directory` | `data/chroma_db` | Diretório de persistência do vector store |
| `temperature` | `0.0` | Determinismo da geração |
| `max_tokens` | `4096` | Orçamento de tokens da resposta |
| `reasoning_effort` | `low` | Reduz tokens de raciocínio interno (modelos gpt-oss) |

Parâmetros de ingestão de CSV (`parametros.csv_base_juridica`):

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `delimitador` | `;` | Separador de colunas do CSV (`;`, `,`, `\t` ou `auto`) |
| `limite_campo_chars` | `500` | Campos com mais caracteres que isso são resumidos antes do embedding |
| `resumo_max_chars` | `500` | Tamanho máximo do resumo gerado para campos grandes |
| `campos_ignorar` | `[]` | Lista de nomes de colunas a ignorar na ingestão |

Parâmetros de tema relevantes ao RAG:

- `dominio_prompt` — preenche o placeholder `{dominio}` do prompt RAG.

Parâmetros de rate limit (`config.rate_limit`) aplicados às chamadas de geração:
`max_retries` (6), `base_delay_s` (2.0), `max_delay_s` (60.0).

---

## 5. Regras de negócio

- **RN-R01 — Mesmo modelo nas duas abordagens:** a LLM RAG e a LLM Padrão usam o
  mesmo modelo de execução; a diferença é somente a presença do contexto.
- **RN-R02 — Resposta restrita ao contexto:** o prompt exige que a resposta use
  **estritamente** as informações do contexto recuperado.
- **RN-R03 — Top-K por configuração:** a recuperação seleciona `top_k` trechos
  (padrão 8), definido em configuração, por MMR (padrão) ou similaridade pura.
- **RN-R04 — Cronometragem separada:** retrieval e geração são medidos
  separadamente; `tempo_total_rag_s = retrieval + geração`.
- **RN-R05 — Backoff fora da métrica:** o tempo de espera por rate limit (429)
  **não** entra na latência medida.
- **RN-R06 — Embeddings locais:** os embeddings são calculados localmente, sem
  custo de API e sem variabilidade de rede.
- **RN-R07 — Persistência e reuso:** o índice é persistido e reutilizado; a
  reindexação só ocorre sob demanda (`forcar=True` / `--reindexar`).
- **RN-R08 — Isolamento por tema:** ao trocar de tema, deve-se trocar o
  `collection_name` ou reindexar, para não misturar embeddings de temas
  diferentes na mesma coleção.
- **RN-R09 — Registro de fontes:** cada resposta RAG registra as fontes
  utilizadas (trecho + metadados), permitindo auditoria e rastreabilidade.
- **RN-R10 — Vector store local:** o `persist_directory` deve permanecer em
  disco local (o ChromaDB usa SQLite + índice; pastas sincronizadas podem causar
  locks/corrupção).

---

## 6. Métricas produzidas pela abordagem RAG

Por pergunta, o pipeline registra:

- `tempo_retrieval_rag_s`, `tempo_geracao_rag_s`, `tempo_total_rag_s`
- `resposta_llm_rag` e `fontes_rag`
- avaliação do Juiz em `avaliacao_llm_rag` (fidelidade, precisão, completude,
  alucinação, fundamentação)
- validação factual dos processos citados em `validacao_processos_rag`

Globalmente (via `utils.calcular_metricas_globais`):

- `tempo_medio_llm_rag_s`, `overhead_medio_rag_s`, `overhead_medio_rag_pct`
- médias das notas RAG e `taxa_alucinacao_rag_pct`

Essas métricas permitem responder à pergunta central do projeto: **o RAG
melhora a qualidade o suficiente para justificar o overhead de latência?**

---

## 7. Limitações da arquitetura atual

- **Recuperação puramente densa:** sem busca lexical (BM25) nem híbrida; termos
  jurídicos exatos (números de artigo, siglas) podem não ser priorizados.
- **MMR sem re-ranking dedicado:** há diversificação por MMR, mas não um
  reordenador (cross-encoder) posterior à recuperação.
- **Passo único:** não há re-recuperação nem refinamento iterativo da consulta.
- **Contexto por stuffing:** com muitos/grandes trechos, pode-se aproximar o
  limite de contexto do modelo.
- **Chunking fixo:** tamanho/overlap fixos, sem segmentação semântica ou por
  estrutura do documento (títulos, artigos).
- **Sem filtro por metadados:** a recuperação não filtra por fonte, data ou tipo
  de documento.
- **Embedding genérico:** `all-MiniLM-L6-v2` é multiuso e não especializado em
  linguagem jurídica em português.

---

## 8. Possibilidades de evolução (experimentos futuros)

As melhorias abaixo são candidatas naturais de teste, aproveitando o fato de o
llm-tester **já medir qualidade e latência** — o que permite comparar cada
variante contra o RAG atual como *baseline*.

### 8.1. Recuperação

- **Busca híbrida (dense + BM25):** combinar similaridade vetorial com busca
  lexical para capturar termos jurídicos exatos.
- **Re-ranking (cross-encoder):** reordenar os candidatos recuperados antes de
  compor o contexto (ex.: modelos de reranking).
- **Ajuste de `top_k` e MMR:** experimentar diferentes `k` e recuperação com
  *Maximal Marginal Relevance* (diversidade vs. redundância).
- **Filtragem por metadados:** restringir por tipo de norma, tribunal ou data.

### 8.2. Indexação e chunking

- **Chunking semântico / estrutural:** segmentar por artigos, incisos e seções,
  em vez de tamanho fixo.
- **Embeddings especializados:** avaliar modelos de embedding treinados em
  português/jurídico.
- **Múltiplas granularidades:** indexar parent/child chunks (small-to-big).

### 8.3. Fluxos avançados de RAG

- **Query expansion / rewriting:** reescrever a pergunta antes de recuperar.
- **Multi-hop / iterativo:** múltiplos ciclos de recuperação para perguntas
  compostas.
- **Self-RAG / correção:** verificar se o contexto suporta a resposta e
  re-recuperar quando necessário.
- **Citação de fontes na resposta:** exigir que o modelo referencie os trechos
  usados (facilita a auditoria e a validação factual).

### 8.4. Compatibilidade com a arquitetura atual

Essas evoluções encaixam nos pontos de extensão já existentes:

- A **recuperação** é isolada em `Pipeline._recuperar_trechos` (troca de
  estratégia: MMR ou similaridade).
- A **geração** é isolada em `Pipeline.executar_rag` (montagem de contexto/prompt).
- Os **parâmetros** são externalizados em `config.json` (novos parâmetros de RAG
  podem ser adicionados sem alterar o fluxo).
- As **métricas e o relatório** já existem, então cada variante pode ser avaliada
  e comparada de forma consistente com o baseline.

> Recomendação metodológica: introduzir **uma variação por vez**, mantendo o
> mesmo dataset, gabarito e modelo, para isolar o efeito de cada mudança sobre
> qualidade e latência.

---

## 9. Rastreabilidade

| Item | Localização |
|------|-------------|
| Ingestão / chunking / indexação | `pipeline.py` → `Pipeline.indexar_base` |
| Loader CSV com resumo de campos | `pipeline.py` → `Pipeline._carregar_csv_base`, `Pipeline._resumir_campo` |
| Cronometragem da indexação | `pipeline.py` → `Pipeline.indexar_base`, `Pipeline._fmt_tempo` |
| Embeddings | `pipeline.py` → `Pipeline._get_embeddings` |
| Recuperação (Top-K + MMR) | `pipeline.py` → `Pipeline._recuperar_trechos` |
| Retrieval + geração + tempos | `pipeline.py` → `Pipeline.executar_rag` |
| Montagem do prompt (placeholders) | `pipeline.py` → `Pipeline._formatar_prompt` |
| Prompt RAG padrão | `config.py` → `DEFAULT_CONFIG["prompts"]["llm_rag"]` |
| Parâmetros de RAG | `config.py` → `DEFAULT_CONFIG["parametros"]`, `config.json` |
| Backoff / rate limit | `utils.py` → `com_backoff` |
| Métricas globais | `utils.py` → `calcular_metricas_globais` |
| Orquestração por pergunta | `pipeline.py` → `executar_benchmarking` |
