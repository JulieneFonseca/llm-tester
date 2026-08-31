# llm-tester

Sistema de **Benchmarking e Avaliação de LLMs (Padrão vs. RAG)** no domínio de
**Direito Previdenciário — Pensão por Morte**.

O `llm-tester` submete um dataset de perguntas jurídicas a uma LLM em duas
abordagens e as compara com um gabarito oficial usando uma **LLM Juiz**:

1. **LLM Padrão** — pergunta enviada diretamente, sem contexto suplementar.
2. **LLM com RAG** — pergunta respondida com contexto recuperado de uma base
   jurídica local (acórdãos/legislação em PDF, TXT ou CSV, indexados em um
   vector store).

Cada etapa é **cronometrada** (`time.perf_counter()`), permitindo medir o
*overhead* de latência introduzido pelo pipeline RAG.

## Documentação

- [Análise de Requisitos](docs/ANALISE_DE_REQUISITOS.md) — requisitos funcionais
  e não funcionais, regras de negócio, casos de uso e rastreabilidade.
- [Especificação da LLM Juiz](docs/LLM_JUIZ.md) — estrutura, configurações,
  rubrica de avaliação e regras de negócio do componente de avaliação
  automatizada.
- [Especificação da LLM RAG](docs/LLM_RAG.md) — arquitetura RAG, tipo de RAG
  utilizado, regras de negócio e possibilidades de evolução para novos testes.

## Stack

- Python 3.10+
- Streamlit (interface web local)
- LangChain + ChromaDB (orquestração RAG e vector store)
- HuggingFace embeddings (`all-MiniLM-L6-v2`, roda local, sem custo)
- Groq API (LLMs — free tier)

## Estrutura

```
llm-tester/
├── data/
│   ├── base_juridica/      # PDFs/TXTs/CSVs (acórdãos, legislação) — vector DB
│   ├── perguntas/
│   │   └── perguntas.json  # Dataset de perguntas de entrada
│   └── gabarito/
│       └── gabarito.json   # Respostas oficiais de referência
├── outputs/                # Relatórios JSON/CSV gerados
├── src/llm_tester/
│   ├── config.py           # Leitura de parâmetros e chave de API
│   ├── pipeline.py         # Ingestão RAG, cronometragem, LLM Padrão e RAG
│   ├── judge.py            # LLM Juiz (avaliação automatizada em JSON)
│   ├── datajud.py          # Validação de processos citados (API DataJud/CNJ)
│   ├── ui.py               # Interface Streamlit (4 abas)
│   └── utils.py            # Arquivos, métricas, backoff, exportações
├── config.json             # Parâmetros padrão
├── app.py                  # Ponto de entrada Streamlit
├── main.py                 # Ponto de entrada CLI (headless)
├── requirements.txt
└── README.md
```

## Instalação

```powershell
pip install -r requirements.txt
copy .env.example .env   # depois edite e coloque sua GROQ_API_KEY
```

Obtenha uma chave gratuita em https://console.groq.com/keys

Opcionalmente, defina `DATAJUD_API_KEY` no `.env` para a validação de processos
(veja a seção *Validação de processos no DataJud*). Se não for definida, o
sistema usa a chave pública do CNJ como *fallback*.

## Uso

### Interface web (Streamlit)

No Windows, basta executar o atalho (cria o ambiente virtual e instala as
dependências na primeira vez):

```powershell
.\executar.bat
```

Ou, manualmente:

```powershell
streamlit run app.py
```

O cabeçalho traz um botão *Sair* (com confirmação) que encerra o servidor local.

Abre 4 abas:
1. **Configurações** — chave de API, upload de perguntas/gabarito/base, seleção
   de modelos e parâmetros (incluindo o *toggle* de validação de processos no
   DataJud).
2. **Dados** — cadastro e manutenção das perguntas e gabaritos usados nas
   execuções.
3. **Execução** — duas sub-abas: **Execução em Lote** (roda as perguntas
   cadastradas selecionadas; o acompanhamento aparece na ordem **console de
   logs** — área rolável com quebra de linha, para ler o conteúdo completo — e
   depois a **tabela** de tempos e notas, ambos atualizados em tempo real) e
   **Execução Individual** (testa uma pergunta avulsa, exibindo as respostas da
   **LLM Padrão** e da **LLM com RAG** lado a lado, com tempos, além do
   detalhamento do Juiz).
4. **Resultados** — **parecer final da LLM Juiz** (abordagem vencedora + texto
   descritivo), KPIs de qualidade e performance, gráficos comparativos
   (Plotly), *Detalhamento da Avaliação do modelo LLM Juiz* (notas
   intermediárias, sinais, justificativa, respostas geradas e validação de
   processos por pergunta), tabela detalhada (critérios em pares Padrão/RAG)
   com filtros e exportação JSON/CSV.

### CLI (headless)

```powershell
python main.py --config config.json
python main.py --config config.json --reindexar
python main.py --modelo openai/gpt-oss-20b
```

## Modelos disponíveis (Groq)

- `openai/gpt-oss-120b` (recomendado para o Juiz)
- `openai/gpt-oss-20b` (execução padrão)
- `groq/compound`
- `groq/compound-mini`
- `qwen/qwen3.6-27b`

> **Modelos de raciocínio (gpt-oss):** os modelos `gpt-oss` separam o raciocínio
> interno do conteúdo final. Com um `max_tokens` baixo, o orçamento pode se
> esgotar no raciocínio e a resposta final voltar vazia. Por isso o `config.json`
> usa `max_tokens: 4096` e `reasoning_effort: "low"` (enviado à API via
> `extra_body`), reduzindo os tokens gastos no raciocínio e deixando espaço para
> a resposta.

## Configuração do RAG

- Chunk size: **1500 tokens** | Overlap: **300 tokens**
- Retriever: **Top-K = 8** trechos, com **MMR** (Maximal Marginal Relevance)
  para diversidade e `mmr_lambda = 0.7`
- Deduplicação automática de chunks idênticos antes da indexação
- Vector store: ChromaDB (persistido em `data/chroma_db`)

A busca na consulta é feita **direto na collection do ChromaDB** (dicionários
crus), evitando falhas de validação quando há registros sem texto no índice, e
descarta trechos vazios automaticamente.

### Formatos aceitos na base jurídica

A base aceita **PDF**, **TXT** e **CSV**. Para CSVs (ex.: exportações de
acórdãos), cada linha vira um documento e campos muito grandes (acima de
`limite_campo_chars`, padrão 500) são **resumidos** automaticamente para gerar
embeddings de melhor qualidade. O **delimitador** do CSV é configurável
(`;` por padrão, também aceita `,`, tab ou detecção automática), tanto no
`config.json` (`parametros.csv_base_juridica`) quanto na aba Configurações.

### Tempos de indexação

A ingestão registra, no log e no relatório JSON (`performance_indexacao`), o
tempo de cada fase: **carga dos documentos**, **chunking + deduplicação** e
**indexação vetorial**, além do total (em segundos e minutos) e do *throughput*
em chunks/s.

### Parecer final da LLM Juiz

Ao término do benchmarking, a LLM Juiz gera um **parecer descritivo
consolidado** que analisa todas as avaliações e **posiciona qual abordagem
(LLM Padrão ou LLM com RAG) teve o melhor desempenho geral**, com a
justificativa das métricas. O parecer aparece no CLI, na aba Resultados, é
salvo no relatório JSON (`execucao_metadata.parecer_final_juiz`) e também numa
seção de resumo ao final do CSV de resultados.

## Configurar outro tema

O tema (domínio) é **configurável** pelo `config.json`, na seção `tema`:

```json
"tema": {
  "nome": "Pensão por Morte",
  "area": "Direito Previdenciário",
  "dominio_prompt": "Direito Previdenciário",
  "especialidade_juiz": "Direito Previdenciário (tema Pensão por Morte)",
  "palavras_chave_datajud": ["pensao", "pensão", "morte", "previdenci", "previdenciário", "beneficio"]
}
```

- `nome` / `area` — usados no cabeçalho da interface, no banner da CLI e no
  metadata dos datasets salvos.
- `dominio_prompt` — preenche o placeholder `{dominio}` dos prompts da LLM
  (Padrão e RAG).
- `especialidade_juiz` — define a especialidade no *system prompt* da LLM Juiz.
- `palavras_chave_datajud` — palavras usadas para validar se o **assunto** do
  processo consultado no DataJud é compatível com o tema.

### Configurar o tema pela interface

Na aba **Configurações** há o painel **🏷️ Tema**, onde é possível editar nome,
área, domínio dos prompts, especialidade do Juiz e as palavras-chave do DataJud.
As alterações são aplicadas **imediatamente à sessão**; para gravar no
`config.json`, use o botão **💾 Salvar tema no config.json** (a chave de API não
é gravada no arquivo, por segurança). Ao mudar o nome do tema, a interface
lembra de **reindexar a base** para não misturar embeddings de temas diferentes.

### Passo a passo para trocar de tema

1. **Carregue os arquivos do novo tema:**
   - Aponte o *Diretório da base jurídica* (aba Configurações) para a pasta com
     os novos PDFs/TXTs.
   - Substitua `perguntas.json` e `gabarito.json` (pela aba **Dados** ou nos
     caminhos definidos em `dados` no `config.json`).
2. **Edite o tema** — pela seção `tema` do `config.json` ou pelo painel
   **🏷️ Tema** na aba Configurações (com o botão de salvar).
3. **Troque o `collection_name`** (em `parametros`) por um nome novo, ou marque
   *Reindexar base do zero* na interface — assim os embeddings do tema antigo
   não se misturam com os do novo no ChromaDB.
4. (Opcional) Se o novo tema **não** for jurídico/federal, desative a validação
   no DataJud (`validar_processos_datajud: false`).

Os prompts usam placeholders (`{dominio}`, `{pergunta}`, `{contexto_recuperado}`),
então a mecânica de execução não muda entre temas.

## Local ou nuvem (armazenamento dos arquivos)

A base jurídica, as perguntas e o gabarito podem ficar em um **diretório local**
ou em uma **pasta de nuvem sincronizada** — desde que o cliente de sincronização
(OneDrive, Google Drive, Dropbox) exponha a pasta como um caminho no sistema de
arquivos. Também funcionam **unidades de rede mapeadas** (`Z:\base`) e caminhos
UNC (`\\servidor\compartilhamento\base`). Basta informar o caminho no campo
*Diretório da base jurídica* da aba **Configurações**.

Todos os acessos usam caminhos do sistema de arquivos (`pathlib.Path`), portanto:

- **Suportado:** `data/base_juridica/` (PDF/TXT/CSV),
  `C:\Users\voce\OneDrive\base`, `Z:\base`, `\\servidor\compartilhamento\base`.
- **Não suportado:** acesso via API/URL (`https://drive.google.com/...`,
  links do SharePoint, buckets S3, `gs://`). Para usar a nuvem, sincronize a
  pasta localmente primeiro.

A interface valida o caminho informado e alerta quando ele está vazio, é uma
URL, não existe ainda ou não contém arquivos PDF/TXT/CSV.

**Cuidados ao usar nuvem sincronizada:**

- **Arquivos "online-only":** OneDrive/Google Drive podem manter arquivos como
  *placeholders* (não baixados). A leitura força o download sob demanda, o que
  pode causar lentidão ou falha se estiver offline. Marque a pasta como *sempre
  disponível neste dispositivo*.
- **Vector store local:** mantenha o `persist_directory` (`data/chroma_db`) em
  disco **local**. O ChromaDB usa SQLite + índice; deixá-lo em uma pasta
  sincronizada pode causar *locks* ou corrupção durante a escrita.

## Métricas de saída

Qualidade (0–5): fidelidade, precisão vs. gabarito, completude; além de taxa de
alucinação e fundamentação correta. Performance: tempo da LLM Padrão, tempo de
retrieval, tempo de geração RAG, tempo total RAG, overhead do RAG e tempo do Juiz.
O relatório também inclui, por pergunta, a validação de processos citados
(`validacao_processos_padrao` / `validacao_processos_rag`); os tempos de
indexação (`performance_indexacao`); e o **parecer final** consolidado da LLM
Juiz (`parecer_final_juiz`), com a abordagem vencedora e o texto descritivo.

## Validação de processos no DataJud

Quando uma resposta (Padrão ou RAG) cita um **número de processo** no padrão
CNJ, o sistema consulta a **API Pública do DataJud (CNJ)** para validar a
citação, funcionando como uma checagem factual antialucinação:

- Verifica se o processo **existe** de fato.
- Recupera o campo **assunto** e valida, por **palavra-chave**, se é compatível
  com o tema da aplicação (Previdenciário / Pensão por Morte).
- Resultado por processo: 🟢 existe e compatível · 🟠 existe mas incompatível ·
  🔴 não encontrado (possível alucinação) · ⚪ fora de escopo/erro.

Os resultados aparecem no **console de logs**, no **relatório** (campos
`validacao_processos_padrao` / `validacao_processos_rag`) e no **dashboard**,
dentro do detalhamento da avaliação do Juiz.

**Escopo atual:** apenas Tribunais Regionais Federais (TRF1–TRF6), onde ficam
os processos previdenciários federais. O TRF é resolvido automaticamente a
partir do próprio número CNJ.

**Chave de API:** definida em `DATAJUD_API_KEY` no `.env`. A chave do DataJud é
pública (fornecida pelo CNJ); se a variável não for definida, o sistema usa a
chave pública publicada como *fallback*.

**Cache:** cada número consultado é memorizado em cache durante a execução,
evitando reconsultar o mesmo processo. Erros de conexão não são cacheados, para
permitir nova tentativa em execuções futuras.

**Ativar/desativar:** controlado pela flag `validar_processos_datajud` (default
`true`) em `parametros` no `config.json`, ou pelo checkbox correspondente na aba
de Configuração da interface. Como a API do DataJud pode ser lenta, desativar
reduz a latência total quando a validação não for necessária.

## Tratamento de rate limit

Chamadas à Groq usam *exponential backoff* automático em caso de HTTP 429.
O tempo de espera do backoff **não** entra na medição de latência das LLMs.
