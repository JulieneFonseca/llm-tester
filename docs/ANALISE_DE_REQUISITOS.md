# Documento de Análise de Requisitos

**Projeto:** llm-tester — Benchmarking e Avaliação de LLMs (Padrão vs. RAG)
**Domínio de referência:** Direito Previdenciário — Pensão por Morte (tema configurável)
**Versão do documento:** 1.0
**Data:** 25/08/2026

---

## 1. Introdução

### 1.1. Objetivo do documento

Este documento descreve os requisitos funcionais e não funcionais do sistema
**llm-tester**, uma ferramenta de benchmarking que compara o desempenho de um
mesmo modelo de linguagem (LLM) em duas abordagens — resposta direta (Padrão) e
resposta com Recuperação Aumentada por Geração (RAG) — sobre um conjunto de
perguntas jurídicas, avaliando os resultados de forma automatizada por meio de
uma LLM Juiz e de uma validação factual externa (API DataJud/CNJ).

### 1.2. Escopo

O sistema submete um dataset de perguntas a uma LLM nas duas abordagens,
cronometra cada etapa, avalia as respostas contra um gabarito oficial e produz
relatórios comparativos (qualidade e performance) em JSON e CSV, além de um
painel visual interativo.

O tema/domínio (por padrão "Pensão por Morte") é **parametrizável**, permitindo
reutilizar o sistema em outros temas jurídicos sem alteração de código.

### 1.3. Definições e siglas

| Termo | Significado |
|-------|-------------|
| **LLM** | Large Language Model (modelo de linguagem) |
| **RAG** | Retrieval-Augmented Generation (geração aumentada por recuperação) |
| **LLM Padrão** | Resposta gerada sem contexto suplementar |
| **LLM com RAG** | Resposta gerada com contexto recuperado da base jurídica |
| **LLM Juiz** | Modelo que avalia automaticamente as respostas |
| **Vector store** | Banco de dados vetorial (ChromaDB) que armazena os embeddings |
| **Chunk** | Trecho de documento resultante da segmentação para indexação |
| **Embedding** | Representação vetorial de um texto |
| **Gabarito** | Resposta oficial de referência |
| **DataJud** | Base Nacional de Dados do Poder Judiciário (CNJ) |
| **CNJ** | Conselho Nacional de Justiça |
| **TRF** | Tribunal Regional Federal |
| **Overhead** | Tempo adicional introduzido pelo pipeline RAG |

---

## 2. Descrição geral

### 2.1. Perspectiva do produto

O llm-tester é uma aplicação local (desktop/servidor pessoal) executada sobre
Python, com duas interfaces de uso: uma web (Streamlit, 4 abas) e uma de linha
de comando (CLI, execução headless). Depende de serviços externos para
inferência das LLMs (Groq API) e para validação factual de processos (API
Pública do DataJud/CNJ). Os embeddings são gerados localmente (HuggingFace),
sem custo e sem dependência de rede.

### 2.2. Funções principais

- Ingestão e indexação vetorial de uma base jurídica (PDF/TXT).
- Execução das duas abordagens (Padrão e RAG) com cronometragem por etapa.
- Avaliação automatizada das respostas por uma LLM Juiz.
- Validação factual de números de processo citados via DataJud.
- Cálculo de métricas de qualidade e performance.
- Visualização de resultados e exportação (JSON/CSV).

### 2.3. Perfis de usuário

| Perfil | Descrição | Uso típico |
|--------|-----------|------------|
| **Pesquisador/Avaliador** | Compara qualidade Padrão vs. RAG | Interface web, análise de resultados |
| **Operador técnico** | Executa em lote / automatiza | CLI headless, integração em scripts |
| **Administrador do tema** | Configura domínio, base, modelos | Aba de Configurações, `config.json` |

### 2.4. Restrições e premissas

- Requer Python 3.10+.
- Requer chave de API da Groq (`GROQ_API_KEY`) para inferência das LLMs.
- A validação DataJud cobre **apenas** Tribunais Regionais Federais (TRF1–TRF6).
- O vector store (ChromaDB) deve residir em disco **local** (evita locks/corrupção).
- Arquivos de base podem estar em pasta de nuvem **sincronizada** (caminho de
  sistema de arquivos), mas **não** via API/URL de nuvem.

---

## 3. Requisitos funcionais

Os requisitos abaixo refletem o comportamento implementado no código.

### 3.1. Configuração e parâmetros

- **RF-01 — Carregar configuração:** o sistema deve ler parâmetros de um
  `config.json` (projeto, tema, API, dados, modelos, parâmetros, rate limit,
  prompts).
- **RF-02 — Chave de API por ambiente:** a `GROQ_API_KEY` deve ser lida de
  variável de ambiente / `.env`, nunca gravada no `config.json`.
- **RF-03 — Tema configurável:** o sistema deve permitir configurar nome, área,
  domínio dos prompts, especialidade do Juiz e palavras-chave do DataJud, tanto
  pelo `config.json` quanto pela interface, com aplicação imediata à sessão e
  opção de gravar no arquivo (sem gravar a chave de API).
- **RF-04 — Seleção de modelos:** o sistema deve permitir escolher o modelo de
  execução e o modelo Juiz dentre a lista de modelos disponíveis (Groq).
- **RF-05 — Parâmetros de execução:** deve permitir ajustar `temperature`,
  `max_tokens`, `reasoning_effort`, `chunk_size`, `chunk_overlap`, `top_k`,
  `embedding_model`, `collection_name` e `persist_directory`.

### 3.2. Dados de teste

- **RF-06 — Carregar perguntas e gabarito:** o sistema deve carregar datasets de
  perguntas e gabarito a partir de arquivos JSON ou CSV (arquivo local ou upload).
- **RF-07 — Manutenção de datasets:** deve permitir cadastrar e editar perguntas
  e gabaritos pela interface, preservando o bloco `metadata` e atualizando o
  `total_perguntas` ao salvar.
- **RF-08 — Indexação por id:** deve associar cada pergunta ao seu gabarito por
  `id` (normalizado como string).

### 3.3. Ingestão e RAG

- **RF-09 — Carregar base jurídica:** deve carregar documentos PDF e TXT de um
  diretório informado (recursivamente), registrando sucesso/erro por arquivo.
- **RF-10 — Segmentação (chunking):** deve segmentar os documentos com
  `chunk_size` (padrão 1000) e `chunk_overlap` (padrão 150).
- **RF-11 — Indexação vetorial:** deve indexar os chunks no ChromaDB, com os
  embeddings gerados localmente (`all-MiniLM-L6-v2` por padrão), persistindo em
  `persist_directory`.
- **RF-12 — Reuso e reindexação:** deve reutilizar um vector store existente e
  permitir forçar a reindexação do zero (flag/`--reindexar`).
- **RF-13 — Recuperação (retrieval):** deve recuperar os `top_k` (padrão 4)
  trechos mais similares à pergunta.

### 3.4. Execução das LLMs

- **RF-14 — LLM Padrão:** deve enviar a pergunta diretamente à LLM, sem contexto,
  usando o prompt configurado com placeholder `{dominio}`.
- **RF-15 — LLM com RAG:** deve compor o prompt com o contexto recuperado e a
  pergunta (placeholders `{contexto_recuperado}` e `{pergunta}`) e gerar a resposta.
- **RF-16 — Registro de fontes:** deve registrar as fontes recuperadas (trecho e
  metadados) associadas à resposta RAG.
- **RF-17 — Execução em lote:** deve executar as perguntas selecionadas em
  sequência, com progresso e logs em tempo real.
- **RF-18 — Execução individual:** deve permitir testar uma pergunta avulsa, com
  gabarito opcional, exibindo as respostas Padrão e RAG lado a lado.

### 3.5. Avaliação (LLM Juiz)

- **RF-19 — Avaliação automatizada:** a LLM Juiz deve receber pergunta, gabarito
  e as duas respostas e retornar avaliação em JSON estrito
  (`response_format=json_object`).
- **RF-20 — Critérios de nota:** deve atribuir, para cada resposta, notas de 0 a
  5 de `nota_fidelidade`, `nota_precisao_gabarito` e `nota_completude`, além dos
  sinais booleanos `alucinacao_detectada` e `fundamentacao_correta` e uma
  `justificativa_juiz`.
- **RF-21 — Especialidade do Juiz:** o system prompt do Juiz deve incorporar a
  especialidade definida no tema.
- **RF-22 — Tolerância a JSON inválido:** deve tratar falhas de parsing do JSON
  do Juiz com fallback (avaliação vazia marcada como alucinação), sem
  interromper a execução.

### 3.6. Validação factual (DataJud)

- **RF-23 — Extração de processos:** deve extrair números de processo no padrão
  CNJ (com máscara ou 20 dígitos) das respostas, sem duplicatas.
- **RF-24 — Consulta ao DataJud:** deve consultar cada processo na API Pública do
  DataJud, resolvendo o TRF (1–6) a partir do próprio número; processos fora do
  escopo (não Justiça Federal) são marcados como fora de escopo.
- **RF-25 — Verificação de existência e assunto:** deve indicar se o processo
  existe e verificar, por palavra-chave, se o assunto é compatível com o tema.
- **RF-26 — Classificação do resultado:** deve classificar cada processo como
  existente e compatível, existente e incompatível, não encontrado (possível
  alucinação) ou fora de escopo/erro.
- **RF-27 — Cache por execução:** deve memorizar consultas bem-sucedidas e
  "não encontrado" em cache; erros de conexão não devem ser cacheados.
- **RF-28 — Ativar/desativar:** deve permitir habilitar/desabilitar a validação
  via `validar_processos_datajud` (padrão ativado) no config ou na interface.
- **RF-29 — Chave DataJud:** deve usar `DATAJUD_API_KEY` do `.env`, com fallback
  para a chave pública do CNJ.

### 3.7. Métricas, relatórios e exportação

- **RF-30 — Cronometragem por etapa:** deve medir tempo da LLM Padrão, do
  retrieval, da geração RAG, do total RAG e da avaliação do Juiz, com
  `time.perf_counter()`.
- **RF-31 — Exclusão do backoff:** o tempo de espera do backoff de rate limit
  **não** deve ser contabilizado na latência das LLMs.
- **RF-32 — Métricas globais:** deve calcular médias de tempo (Padrão, RAG),
  overhead médio (s e %), médias das três notas por abordagem e taxas de
  alucinação (%) por abordagem.
- **RF-33 — Relatório estruturado:** deve montar um relatório com metadados de
  execução (data/hora, modelos, total), performance temporal global, métricas de
  qualidade global e resultados detalhados por pergunta.
- **RF-34 — Exportação:** deve exportar o relatório em JSON e em CSV (uma linha
  por pergunta, critérios em pares Padrão/RAG) no diretório de saída.
- **RF-35 — Painel de resultados:** deve exibir KPIs, gráficos comparativos de
  qualidade e de performance temporal, detalhamento da avaliação do Juiz e uma
  tabela detalhada com filtros (ex.: só casos com alucinação; só quando RAG >
  Padrão em precisão).

### 3.8. Interface e controle de execução

- **RF-36 — Interface web em abas:** Configurações, Dados, Execução (lote e
  individual) e Resultados.
- **RF-37 — CLI headless:** deve permitir executar via `main.py` com `--config`,
  `--reindexar` e `--modelo`.
- **RF-38 — Logs em tempo real:** deve exibir o andamento (por pergunta, etapas,
  notas do Juiz, validações DataJud e eventos de rate limit).
- **RF-39 — Seleção de perguntas em lote:** deve oferecer "Selecionar todas" e
  seleção individual sincronizadas (desmarcar um item desmarca "Selecionar
  todas"; marcar todos os itens remarca).
- **RF-40 — Encerrar servidor:** a interface deve oferecer um controle para
  encerrar o servidor local.
- **RF-41 — Validação de diretório:** deve validar o caminho da base e alertar
  quando vazio, quando for URL, inexistente ou sem arquivos PDF/TXT.

---

## 4. Requisitos não funcionais

### 4.1. Desempenho

- **RNF-01:** a cronometragem deve isolar a latência real das LLMs, excluindo o
  backoff (ver RF-31).
- **RNF-02:** os embeddings devem ser calculados localmente, sem chamadas de
  rede, para reduzir custo e latência variável.
- **RNF-03:** a validação DataJud deve ser opcional para não penalizar a latência
  total quando não necessária.

### 4.2. Confiabilidade e resiliência

- **RNF-04:** chamadas à Groq devem usar exponential backoff com jitter em caso
  de HTTP 429, respeitando `max_retries`, `base_delay_s` e `max_delay_s`.
- **RNF-05:** falhas de leitura de documentos individuais não devem abortar a
  ingestão (registrar erro e continuar).
- **RNF-06:** a consulta ao DataJud deve tolerar instabilidade (uma tentativa
  extra em timeout) e degradar graciosamente registrando o erro por processo.
- **RNF-07:** falha de parsing do JSON do Juiz não deve interromper a execução.

### 4.3. Segurança

- **RNF-08:** segredos (`GROQ_API_KEY`) devem vir do ambiente e nunca ser
  persistidos no `config.json` ou em arquivos versionados.
- **RNF-09:** o `.env` deve estar no `.gitignore`.

### 4.4. Usabilidade

- **RNF-10:** a interface deve ser acessível localmente pelo navegador e
  fornecer feedback em tempo real (logs, progresso, tabelas dinâmicas).
- **RNF-11:** mensagens de validação e alertas devem ser claras (ex.: chave
  ausente, diretório inválido, necessidade de reindexar ao trocar de tema).

### 4.5. Manutenibilidade e portabilidade

- **RNF-12:** o domínio deve ser parametrizável via `config.json`/interface sem
  alterar código, por meio de prompts com placeholders.
- **RNF-13:** bibliotecas pesadas devem ser importadas de forma tardia (lazy
  import) para não onerar a carga do módulo.
- **RNF-14:** os acessos a arquivos devem usar caminhos de sistema de arquivos
  (`pathlib`), suportando diretório local, nuvem sincronizada, unidade de rede
  mapeada e caminho UNC.

### 4.6. Custo

- **RNF-15:** o sistema deve operar com o *free tier* da Groq e embeddings
  locais, mantendo o custo próximo de zero.

---

## 5. Dependências externas

| Dependência | Papel | Observação |
|-------------|-------|------------|
| **Groq API** | Inferência das LLMs (execução e Juiz) | Requer `GROQ_API_KEY` |
| **API DataJud (CNJ)** | Validação factual de processos | Escopo TRF1–TRF6; chave pública |
| **ChromaDB** | Vector store | Persistência local |
| **HuggingFace Embeddings** | Geração de embeddings | `all-MiniLM-L6-v2`, local |
| **LangChain** | Orquestração RAG (loaders, splitter, retriever) | — |
| **Streamlit** | Interface web | — |
| **Plotly / Pandas** | Gráficos e tabelas | — |

---

## 6. Contrato de saída (relatório)

O relatório final contém:

- `execucao_metadata`: `data_hora`, `modelo_testado`, `modelo_juiz`,
  `total_perguntas`, `performance_temporal_global` e `metricas_qualidade_global`.
- `resultados_detalhados`: por pergunta — `id_pergunta`, `pergunta`,
  `gabarito_oficial`, respostas Padrão e RAG, tempos (padrão, retrieval, geração,
  total RAG, juiz), `fontes_rag`, `avaliacao_llm_padrao`, `avaliacao_llm_rag` e
  `validacao_processos_padrao` / `validacao_processos_rag`.

Exportações disponíveis: **JSON** (estrutura completa) e **CSV** (uma linha por
pergunta, critérios em pares Padrão/RAG).

---

## 7. Regras de negócio

- **RN-01:** a mesma LLM é usada nas duas abordagens; a diferença é a presença ou
  não de contexto recuperado.
- **RN-02:** a nota de qualidade varia de 0 a 5 conforme a escala definida no
  system prompt do Juiz.
- **RN-03:** um processo citado e não encontrado no DataJud é sinalizado como
  possível alucinação.
- **RN-04:** a compatibilidade de tema do processo é recalculada conforme as
  palavras-chave do tema vigente, mesmo para itens em cache.
- **RN-05:** ao trocar o nome do tema, recomenda-se trocar `collection_name` ou
  reindexar a base, para não misturar embeddings de temas distintos.
- **RN-06:** apenas processos da Justiça Federal (J=4) e TRFs 01–06 são
  validáveis; os demais ficam fora de escopo.

---

## 8. Casos de uso principais

1. **Executar benchmarking em lote:** configurar API/modelos/tema → carregar
   perguntas e gabarito → indexar base → executar seleção → visualizar
   resultados → exportar JSON/CSV.
2. **Testar pergunta individual:** informar pergunta (e gabarito opcional) →
   obter respostas Padrão e RAG lado a lado com tempos e avaliação do Juiz.
3. **Trocar de tema:** apontar nova base, substituir perguntas/gabarito, editar
   o tema, reindexar (ou trocar `collection_name`) e, se aplicável, desativar a
   validação DataJud.
4. **Execução headless (CLI):** rodar `python main.py --config config.json`
   (opcionalmente `--reindexar` e `--modelo`) para gerar relatórios sem interface.

---

## 9. Itens fora de escopo (versão atual)

- Validação de processos fora da Justiça Federal (estaduais, trabalhistas, etc.).
- Acesso a arquivos de nuvem via API/URL (somente pasta sincronizada localmente).
- Fine-tuning de modelos.
- Autenticação multiusuário / controle de acesso (ferramenta local, mono-usuário).

---

## 10. Rastreabilidade (requisito → implementação)

| Requisito | Módulo/Arquivo principal |
|-----------|--------------------------|
| RF-01 a RF-05 | `config.py`, `config.json`, `ui.py` |
| RF-06 a RF-08 | `utils.py` (carregar/salvar/indexar), `ui.py` |
| RF-09 a RF-13 | `pipeline.py` (`indexar_base`, `_get_retriever`) |
| RF-14 a RF-18 | `pipeline.py` (`executar_padrao`, `executar_rag`), `ui.py` |
| RF-19 a RF-22 | `judge.py` |
| RF-23 a RF-29 | `datajud.py` |
| RF-30 a RF-35 | `pipeline.py`, `utils.py` (métricas/relatório), `ui.py` |
| RF-36 a RF-41 | `ui.py`, `main.py` |
| RNF-04 (backoff) | `utils.py` (`com_backoff`) |
