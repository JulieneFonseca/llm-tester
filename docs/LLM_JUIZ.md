# Documento de Especificação — LLM Juiz

**Projeto:** llm-tester
**Componente:** LLM Juiz (avaliação automatizada)
**Arquivo-fonte:** `src/llm_tester/judge.py`
**Versão do documento:** 1.1
**Data:** 31/08/2026

---

## 1. Visão geral

A **LLM Juiz** é o componente responsável pela **avaliação automatizada** das
respostas produzidas pelo sistema. Para cada pergunta, ela recebe:

- a **pergunta** original;
- o **gabarito oficial** de referência;
- a **resposta da LLM Padrão** (sem contexto);
- a **resposta da LLM com RAG** (com contexto jurídico recuperado);

e devolve, em **JSON estrito**, uma avaliação comparativa de qualidade jurídica
das duas respostas, com notas, sinais booleanos e justificativa.

O objetivo é substituir a avaliação manual por um julgamento consistente,
reproduzível e estruturado, permitindo comparar objetivamente as abordagens
Padrão e RAG.

---

## 2. Estrutura do componente

O Juiz é implementado na classe `Judge` (`src/llm_tester/judge.py`), composta
pelos seguintes elementos:

| Elemento | Responsabilidade |
|----------|------------------|
| `SYSTEM_PROMPT_JUIZ_TEMPLATE` | Template do *system prompt* com a rubrica de avaliação |
| `Judge.__init__` | Inicializa modelo, rate limit e monta o system prompt a partir do tema |
| `Judge._get_llm` | Instancia o cliente da LLM (lazy) com `response_format` JSON |
| `Judge._get_llm_texto` | Instancia o cliente da LLM (lazy) para saída em texto livre (parecer final) |
| `Judge.avaliar` | Executa a avaliação por pergunta e cronometra o tempo do Juiz |
| `Judge.parecer_final` | Gera o parecer descritivo consolidado (Padrão vs. RAG) |
| `Judge._parse_json` | Faz o parsing tolerante do JSON retornado |
| `_avaliacao_vazia` | Fallback usado quando o parsing falha |

### 2.1. Fluxo de execução

```
avaliar(pergunta, gabarito, resposta_padrao, resposta_rag)
        │
        ├── monta mensagem (pergunta + gabarito + resposta da LLM Padrão e da LLM com RAG)
        ├── SystemMessage (system prompt do Juiz) + HumanMessage (mensagem acima)
        ├── invoca a LLM via com_backoff() ── cronometra só a chamada real
        ├── _parse_json(conteúdo) ── com fallback tolerante
        └── retorna avaliação + tempo_avaliacao_juiz_s
```

A chamada é envolvida por `com_backoff()` (de `utils.py`) para tratar rate limit
(HTTP 429). **A cronometragem (`time.perf_counter()`) mede apenas a execução real
da LLM**, excluindo o tempo de espera do backoff.

---

## 3. Configurações

### 3.1. Modelo do Juiz

- Definido por `config.modelos["llm_juiz"]` (padrão: `openai/gpt-oss-120b`).
- Pode ser sobrescrito via parâmetro `modelo_juiz` na construção do `Judge` ou na
  chamada de `executar_benchmarking(...)`, e pela interface na aba de
  Configurações.
- O modelo de 120b é recomendado como Juiz por ter maior capacidade de análise.

### 3.2. Parâmetros do cliente LLM

O cliente é instanciado (via `ChatGroq`) com:

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `model` | modelo do Juiz | Modelo avaliador configurado |
| `temperature` | `0.0` | Determinismo — avaliações estáveis e reproduzíveis |
| `api_key` | `GROQ_API_KEY` | Autenticação (ambiente/.env) |
| `max_retries` | `0` | O retry é tratado manualmente por `com_backoff` (fora da cronometragem) |
| `model_kwargs` | `{"response_format": {"type": "json_object"}}` | Força saída em JSON parseável |

Para o **parecer final** (texto livre), o Juiz usa uma segunda instância
(`_get_llm_texto`) **sem** `response_format` JSON e com `temperature = 0.2`,
permitindo uma redação mais fluida do parecer descritivo.

### 3.3. Especialidade (tema)

O *system prompt* do Juiz é montado a partir da especialidade definida no tema:

- Origem: `config.tema["especialidade_juiz"]`
  (padrão: `"Direito Previdenciário (tema Pensão por Morte)"`).
- A montagem usa `str.replace("{especialidade}", ...)` — **e não** `str.format` —
  porque o template contém chaves `{}` do próprio exemplo de JSON, que não devem
  ser interpretadas como placeholders.

### 3.4. Rate limit (backoff)

Herdado de `config.rate_limit`:

| Parâmetro | Padrão |
|-----------|--------|
| `max_retries` | `6` |
| `base_delay_s` | `2.0` |
| `max_delay_s` | `60.0` |

Backoff exponencial com *jitter*, acionado em HTTP 429.

---

## 4. Rubrica de avaliação (system prompt)

O Juiz avalia **cada uma das duas respostas** segundo os critérios abaixo.

### 4.1. Critérios de nota (escala 0 a 5)

| Campo | Significado |
|-------|-------------|
| `nota_fidelidade` | Fidelidade às fontes jurídicas / ausência de invenção de conteúdo |
| `nota_precisao_gabarito` | Proximidade da resposta em relação ao gabarito oficial |
| `nota_completude` | Cobertura dos pontos essenciais do gabarito |

### 4.2. Sinais booleanos

| Campo | Significado |
|-------|-------------|
| `alucinacao_detectada` | `true` se a resposta contém informação incorreta ou não suportada (ex.: jurisprudência inexistente, dispositivo legal errado) |
| `fundamentacao_correta` | `true` se cita/fundamenta corretamente (súmula, artigo de lei, acórdão) de forma condizente com o gabarito |

### 4.3. Justificativa

| Campo | Significado |
|-------|-------------|
| `justificativa_juiz` | Explicação objetiva e curta da avaliação |

### 4.4. Escala de notas (0 a 5)

| Nota | Interpretação |
|------|---------------|
| 0 | Totalmente incorreto |
| 1 | Majoritariamente incorreto |
| 2 | Parcialmente correto |
| 3 | Correto nos pontos principais |
| 4 | Correto e completo |
| 5 | Totalmente fiel e preciso |

---

## 5. Contrato de entrada e saída

### 5.1. Entrada (mensagem do usuário)

A mensagem enviada ao Juiz contém quatro blocos rotulados:

- `## PERGUNTA:` — a pergunta avaliada.
- `## GABARITO OFICIAL:` — a resposta de referência.
- `## RESPOSTA DA LLM PADRÃO (sem contexto):` — resposta da abordagem Padrão.
- `## RESPOSTA DA LLM COM RAG (com contexto jurídico):` — resposta da abordagem RAG.

O Juiz é instruído a avaliar a **resposta da LLM Padrão** em
`avaliacao_llm_padrao` e a **resposta da LLM com RAG** em `avaliacao_llm_rag`.
Nas justificativas, o Juiz é orientado a **sempre** se referir às respostas
como "a resposta da LLM Padrão" e "a resposta da LLM com RAG" (nunca "A"/"B"),
tornando o parecer mais legível.

### 5.2. Saída (JSON estrito)

```json
{
  "avaliacao_llm_padrao": {
    "nota_fidelidade": 0-5,
    "nota_precisao_gabarito": 0-5,
    "nota_completude": 0-5,
    "alucinacao_detectada": true|false,
    "fundamentacao_correta": true|false,
    "justificativa_juiz": "<texto>"
  },
  "avaliacao_llm_rag": {
    "nota_fidelidade": 0-5,
    "nota_precisao_gabarito": 0-5,
    "nota_completude": 0-5,
    "alucinacao_detectada": true|false,
    "fundamentacao_correta": true|false,
    "justificativa_juiz": "<texto>"
  }
}
```

Além dos dois blocos, o método `avaliar` acrescenta ao retorno o campo
`tempo_avaliacao_juiz_s` (float, segundos), medido separadamente.

---

## 5-A. Parecer final consolidado (`Judge.parecer_final`)

Após a avaliação de **todas** as perguntas, o Juiz produz um **parecer
descritivo final** que consolida os resultados e **posiciona qual abordagem
(LLM Padrão ou LLM com RAG) teve o melhor desempenho geral**.

### 5-A.1. Entrada

- **Métricas globais** (`utils.calcular_metricas_globais`): médias de fidelidade,
  precisão, completude, taxa de alucinação e tempos por abordagem, além do
  overhead do RAG.
- **Resumo por pergunta:** notas e sinais (alucinação) de cada abordagem, caso a
  caso.

### 5-A.2. Saída

Diferente da avaliação por pergunta, o parecer é **texto livre** (não JSON).
O método retorna:

```json
{
  "abordagem_vencedora": "RAG | Padrão | Empate",
  "parecer_texto": "<parecer descritivo, 3 a 5 parágrafos>",
  "tempo_parecer_juiz_s": <float>
}
```

O texto começa com uma linha no formato `VENCEDORA: <RAG|Padrão|Empate>`, da
qual é extraído o campo `abordagem_vencedora`.

### 5-A.3. Conteúdo do parecer

O Juiz é instruído a cobrir, em texto corrido:

1. Qual abordagem teve **melhor desempenho geral** e por quê (citando métricas).
2. **Pontos fortes e fracos** de cada abordagem observados nos casos.
3. A questão central: o ganho de qualidade do RAG **justifica o overhead** de
   latência?
4. Uma **recomendação prática** de qual abordagem usar no domínio.

### 5-A.4. Destino

O parecer é gravado em `execucao_metadata.parecer_final_juiz` no relatório JSON
e exibido no **CLI** (`main.py`) e na aba **Resultados** da interface.

---

## 6. Regras de negócio

- **RN-J01 — Avaliação comparativa única:** as duas respostas (Padrão e RAG) são
  avaliadas na **mesma** chamada, contra o **mesmo** gabarito, garantindo
  critério idêntico entre as abordagens.
- **RN-J02 — Determinismo:** `temperature = 0.0` para tornar a avaliação estável
  e reproduzível.
- **RN-J03 — Saída sempre parseável:** o Juiz é forçado a produzir JSON
  (`response_format=json_object`).
- **RN-J04 — Escala fixa 0–5:** todas as notas seguem a escala definida na
  rubrica; valores fora dela não são esperados.
- **RN-J05 — Fidelidade x fundamentação:** `nota_fidelidade` mede ausência de
  invenção; `fundamentacao_correta` verifica se a citação legal/jurisprudencial
  está correta e condizente com o gabarito. São sinais distintos e complementares.
- **RN-J06 — Especialidade orientada ao tema:** a especialidade do Juiz é
  derivada do tema configurado, alinhando o julgamento ao domínio em avaliação.
- **RN-J07 — Backoff fora da métrica:** o tempo de espera por rate limit não é
  contabilizado no `tempo_avaliacao_juiz_s`.
- **RN-J08 — Independência do DataJud:** o Juiz produz seu julgamento sem
  depender da validação de processos no DataJud; a validação factual é uma
  checagem **complementar e separada** (módulo `datajud.py`).
- **RN-J09 — Nomenclatura das respostas:** nas justificativas e no parecer, as
  abordagens são referidas como "a resposta da LLM Padrão" e "a resposta da LLM
  com RAG" — nunca "resposta A/B".
- **RN-J10 — Parecer não bloqueante:** a geração do parecer final é
  **tolerante a falhas**: se a chamada falhar, o benchmarking conclui
  normalmente e o parecer fica vazio no relatório (não interrompe a execução).

---

## 7. Tratamento de erros e resiliência

### 7.1. Parsing tolerante (`_parse_json`)

1. Tenta `json.loads(content)` diretamente.
2. Em caso de falha, tenta extrair o maior objeto entre a primeira `{` e a
   última `}` do conteúdo e parseá-lo.
3. Se ainda assim falhar, retorna `{}` e prossegue para o fallback.

### 7.2. Fallback por bloco ausente (`_avaliacao_vazia`)

Se `avaliacao_llm_padrao` e/ou `avaliacao_llm_rag` não estiverem presentes no
JSON parseado, o bloco ausente é preenchido com uma **avaliação vazia**:

```json
{
  "nota_fidelidade": 0,
  "nota_precisao_gabarito": 0,
  "nota_completude": 0,
  "alucinacao_detectada": true,
  "fundamentacao_correta": false,
  "justificativa_juiz": "Falha ao parsear avaliação do juiz."
}
```

Consequências dessa regra:

- **A execução nunca é interrompida** por um retorno malformado do Juiz.
- Uma falha de parsing é tratada de forma **conservadora**: nota zero e
  `alucinacao_detectada = true`, sinalizando explicitamente o problema na
  justificativa.

### 7.3. Rate limit

Erros 429 acionam `com_backoff` (exponencial + jitter), respeitando os limites
de `config.rate_limit`. Após esgotar `max_retries`, a exceção é propagada.

---

## 8. Integração no pipeline

No `pipeline.executar_benchmarking`, para cada pergunta:

1. Obtêm-se as respostas Padrão e RAG.
2. Chama-se `juiz.avaliar(...)` com pergunta, gabarito e as duas respostas.
3. As notas e sinais são registrados no log (por abordagem) e incluídos no
   resultado da pergunta nos campos `avaliacao_llm_padrao`, `avaliacao_llm_rag`
   e `tempo_avaliacao_juiz_s`.

Esses campos alimentam:

- as **métricas globais** (médias de notas por abordagem e taxa de alucinação —
  ver `utils.calcular_metricas_globais`);
- o **relatório** JSON/CSV;
- o **dashboard** de resultados (KPIs, gráficos e detalhamento da avaliação).

Ao final de todas as perguntas, `executar_benchmarking` chama
`juiz.parecer_final(...)` com as métricas globais e os resultados, e grava o
retorno em `execucao_metadata.parecer_final_juiz` no relatório.

---

## 9. Rastreabilidade

| Item | Localização |
|------|-------------|
| Template do system prompt / rubrica | `judge.py` → `SYSTEM_PROMPT_JUIZ_TEMPLATE` |
| Configuração do cliente LLM (JSON) | `judge.py` → `Judge._get_llm` |
| Configuração do cliente LLM (texto) | `judge.py` → `Judge._get_llm_texto` |
| Montagem por especialidade (tema) | `judge.py` → `Judge.__init__` |
| Execução e cronometragem (por pergunta) | `judge.py` → `Judge.avaliar` |
| Parecer final consolidado | `judge.py` → `Judge.parecer_final` |
| Parsing tolerante e fallback | `judge.py` → `Judge._parse_json`, `_avaliacao_vazia` |
| Backoff / rate limit | `utils.py` → `com_backoff` |
| Agregação das notas | `utils.py` → `calcular_metricas_globais` |
| Integração no fluxo | `pipeline.py` → `executar_benchmarking` |
| Defaults de modelo/tema/rate | `config.py` → `DEFAULT_CONFIG` |
