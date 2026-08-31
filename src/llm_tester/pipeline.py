"""
Orquestração do pipeline de benchmarking.

Responsável por:
  - Ingestão da base jurídica (chunking + indexação vetorial no ChromaDB)
  - Chamada à LLM Padrão (sem contexto) com cronometragem
  - Recuperação de contexto (retrieval) + chamada à LLM com RAG, cronometradas
    separadamente com time.perf_counter()

A cronometragem exclui o tempo de espera do backoff de rate limit.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .utils import com_backoff


class Pipeline:
    """Executa as duas abordagens (Padrão e RAG) medindo a latência de cada etapa."""

    def __init__(self, config: Config, modelo_execucao: str | None = None):
        self.config = config
        self.params = config.parametros
        self.prompts = config.prompts
        self.rate = config.rate_limit
        self.modelo = modelo_execucao or config.modelos["llm_execucao"]
        # Domínio do tema, injetado no placeholder {dominio} dos prompts.
        self.dominio = config.tema.get("dominio_prompt", "Direito Previdenciário")

        self._llm = None
        self._vector_store = None
        self._embeddings = None

        # Preenchido por indexar_base() com os tempos de cada fase da indexação.
        self.tempos_indexacao: dict[str, Any] = {}

    # -- Lazy imports para não exigir libs pesadas na importação do módulo ----

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq

            # reasoning_effort reduz os tokens gastos no raciocínio interno dos
            # modelos gpt-oss, evitando que o orçamento de max_tokens seja todo
            # consumido antes da resposta final (content vazio).
            # Enviado via extra_body para que vá no corpo da requisição à API,
            # em vez de ser passado como kwarg ao cliente Groq (que não o aceita).
            model_kwargs = {}
            reasoning_effort = self.params.get("reasoning_effort")
            if reasoning_effort:
                model_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

            self._llm = ChatGroq(
                model=self.modelo,
                temperature=self.params.get("temperature", 0.0),
                max_tokens=self.params.get("max_tokens", 4096),
                api_key=self.config.groq_api_key,
                max_retries=0,  # backoff é tratado manualmente (fora da cronometragem)
                model_kwargs=model_kwargs,
            )
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.params.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        return self._embeddings

    # -- Ingestão RAG --------------------------------------------------------

    def _carregar_csv_base(
        self,
        caminho: Path,
        log: Callable[[str], None],
    ) -> list:
        """
        Carrega um CSV da base jurídica e retorna Documents do LangChain.

        Lógica genérica para campos grandes:
          - Campos com conteúdo > `csv_limite_campo_chars` (padrão 500) são
            truncados inteligentemente para um resumo de até
            `csv_resumo_max_chars` (padrão 500) caracteres, preservando as
            primeiras sentenças completas que cabem nesse limite.
          - Cada linha do CSV vira um Document cujo `page_content` é a
            concatenação dos campos (com labels) e cujos `metadata` guardam
            os valores originais das colunas curtas + nome do arquivo.
        """
        import csv as csv_mod
        from langchain_core.documents import Document

        csv_config = self.params.get("csv_base_juridica", {})
        limite_campo = csv_config.get("limite_campo_chars", 500)
        resumo_max = csv_config.get("resumo_max_chars", 500)
        campos_ignorar = set(csv_config.get("campos_ignorar", []))
        # Delimitador do CSV. Padrão ';' porque as ementas contêm muitas
        # vírgulas no texto, e a extração usa ';' como separador de colunas.
        # Use "auto" para detectar automaticamente entre ';', ',' e tab.
        delim_config = csv_config.get("delimitador", ";")

        with open(caminho, "r", encoding="utf-8-sig") as f:
            amostra = f.read(4096)
            f.seek(0)
            if delim_config == "auto":
                delimitador = ";"
                try:
                    dialect = csv_mod.Sniffer().sniff(amostra, delimiters=",;\t")
                    delimitador = dialect.delimiter
                except csv_mod.Error:
                    # Fallback: escolhe o separador mais frequente na 1ª linha.
                    primeira_linha = amostra.splitlines()[0] if amostra else ""
                    delimitador = max(
                        [",", ";", "\t"],
                        key=lambda d: primeira_linha.count(d),
                    )
            else:
                delimitador = delim_config
            reader = csv_mod.DictReader(f, delimiter=delimitador)
            linhas = list(reader)

        if not linhas:
            return []

        documentos = []
        campos_resumidos_log: set[str] = set()

        for idx, row in enumerate(linhas):
            partes_conteudo: list[str] = []
            metadata: dict[str, Any] = {"source": str(caminho), "csv_row": idx}

            for campo, valor in row.items():
                # Campos extras (linha com mais colunas que o cabeçalho) vêm
                # sob a chave None como lista — nome de coluna inválido, ignora.
                if campo is None:
                    continue
                if valor is None or campo in campos_ignorar:
                    continue

                # DictReader pode devolver lista quando há colunas a mais na
                # linha; normaliza qualquer valor para string única.
                if isinstance(valor, (list, tuple)):
                    valor = " ".join(str(v) for v in valor if v is not None)
                else:
                    valor = str(valor)

                valor_limpo = valor.strip()
                if not valor_limpo:
                    continue
                if len(valor_limpo) > limite_campo:
                    # Resumo inteligente: preserva sentenças completas até o limite
                    valor_resumido = self._resumir_campo(valor_limpo, resumo_max)
                    partes_conteudo.append(f"{campo}: {valor_resumido}")
                    campos_resumidos_log.add(campo)
                else:
                    partes_conteudo.append(f"{campo}: {valor_limpo}")
                    # Campos curtos vão para metadata (facilitam filtragem)
                    if len(valor_limpo) <= 200:
                        metadata[campo] = valor_limpo

            if partes_conteudo:
                doc = Document(
                    page_content="\n".join(partes_conteudo),
                    metadata=metadata,
                )
                documentos.append(doc)

        if campos_resumidos_log:
            log(
                f"   [csv] Campos resumidos (>{limite_campo} chars -> "
                f"<={resumo_max} chars): {', '.join(sorted(campos_resumidos_log))}"
            )

        return documentos

    @staticmethod
    def _resumir_campo(texto: str, max_chars: int) -> str:
        """
        Trunca texto longo preservando sentenças completas até `max_chars`.

        Estratégia:
          1. Divide o texto por finais de sentença (. ! ?)
          2. Acumula sentenças enquanto couberem no limite
          3. Se nem a primeira sentença couber, corta no último espaço antes do
             limite e adiciona '…'
        """
        if len(texto) <= max_chars:
            return texto

        # Tenta preservar sentenças completas
        import re
        sentencas = re.split(r'(?<=[.!?])\s+', texto)
        acumulado = ""
        for sentenca in sentencas:
            candidato = (acumulado + " " + sentenca).strip() if acumulado else sentenca
            if len(candidato) <= max_chars:
                acumulado = candidato
            else:
                break

        if acumulado and len(acumulado) >= 100:
            return acumulado

        # Fallback: corta no último espaço antes do limite
        corte = texto[:max_chars]
        ultimo_espaco = corte.rfind(" ")
        if ultimo_espaco > max_chars // 2:
            return corte[:ultimo_espaco] + "…"
        return corte + "…"

    @staticmethod
    def _fmt_tempo(segundos: float) -> str:
        """
        Formata duração em segundos. Para valores >= 60s, acrescenta a
        conversão em minutos (ex.: '1065.0s (17.8 min)').
        """
        if segundos >= 60:
            return f"{segundos:.1f}s ({segundos / 60:.1f} min)"
        return f"{segundos:.1f}s"

    def indexar_base(
        self,
        diretorio: str,
        log: Callable[[str], None] | None = None,
        forcar: bool = False,
    ) -> int:
        """
        Carrega documentos (PDF, TXT, CSV), aplica chunking e indexa no ChromaDB.
        Retorna o número de chunks indexados.
        """
        log = log or (lambda m: None)
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        persist_dir = self.params.get("persist_directory", "data/chroma_db")
        collection = self.params.get("collection_name", "pensao_por_morte")

        if not forcar and Path(persist_dir).exists():
            t0 = time.perf_counter()
            log("[base] Vector store existente encontrado. Carregando...")
            self._vector_store = Chroma(
                collection_name=collection,
                embedding_function=self._get_embeddings(),
                persist_directory=persist_dir,
            )
            n_chunks = self._vector_store._collection.count()
            self.tempos_indexacao = {
                "reindexado": False,
                "total_chunks": n_chunks,
                "tempo_carga_documentos_s": 0.0,
                "tempo_chunking_s": 0.0,
                "tempo_indexacao_vetorial_s": 0.0,
                "tempo_total_indexacao_s": round(time.perf_counter() - t0, 2),
            }
            return n_chunks

        base = Path(diretorio)
        if not base.exists():
            raise FileNotFoundError(f"Base jurídica não encontrada: {diretorio}")

        t_inicio = time.perf_counter()

        documentos = []
        arquivos_doc = list(base.glob("**/*.pdf")) + list(base.glob("**/*.txt"))
        arquivos_csv = list(base.glob("**/*.csv"))
        total_arquivos = len(arquivos_doc) + len(arquivos_csv)
        log(f"[docs] {total_arquivos} arquivo(s) na base jurídica "
            f"({len(arquivos_doc)} PDF/TXT + {len(arquivos_csv)} CSV).")

        t_carga_ini = time.perf_counter()

        # Carregar PDFs e TXTs
        for arq in arquivos_doc:
            try:
                if arq.suffix.lower() == ".pdf":
                    documentos.extend(PyPDFLoader(str(arq)).load())
                else:
                    documentos.extend(
                        TextLoader(str(arq), encoding="utf-8").load()
                    )
                log(f"   [ok] {arq.name}")
            except Exception as exc:  # noqa: BLE001
                log(f"   [erro] Falha ao ler {arq.name}: {exc}")

        # Carregar CSVs (com resumo automático de campos grandes)
        for arq in arquivos_csv:
            try:
                docs_csv = self._carregar_csv_base(arq, log)
                documentos.extend(docs_csv)
                log(f"   [ok] {arq.name} ({len(docs_csv)} registros)")
            except Exception as exc:  # noqa: BLE001
                log(f"   [erro] Falha ao ler {arq.name}: {exc}")

        if not documentos:
            raise ValueError(
                f"Nenhum documento carregado de '{diretorio}'. "
                "Adicione PDFs, TXTs ou CSVs à base jurídica."
            )

        tempo_carga = time.perf_counter() - t_carga_ini
        log(f"[tempo] Carga dos documentos: {self._fmt_tempo(tempo_carga)} "
            f"({len(documentos)} documentos).")

        t_chunk_ini = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.params.get("chunk_size", 1000),
            chunk_overlap=self.params.get("chunk_overlap", 150),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documentos)
        log(
            f"[chunks] {len(chunks)} chunks "
            f"(size={self.params.get('chunk_size')}, "
            f"overlap={self.params.get('chunk_overlap')})"
        )

        # Sanitização + deduplicação:
        #  - descarta chunks sem conteúdo (page_content None ou vazio), que
        #    quebrariam a validação do Document/ChromaDB;
        #  - remove chunks com conteúdo idêntico (mantém o primeiro).
        vistos: set[str] = set()
        chunks_unicos = []
        n_vazios = 0
        for chunk in chunks:
            conteudo = chunk.page_content
            if conteudo is None:
                n_vazios += 1
                continue
            texto_norm = str(conteudo).strip()
            if not texto_norm:
                n_vazios += 1
                continue
            # Garante que o page_content seja sempre string válida.
            chunk.page_content = str(conteudo)
            if texto_norm not in vistos:
                vistos.add(texto_norm)
                chunks_unicos.append(chunk)
        n_duplicados = len(chunks) - len(chunks_unicos) - n_vazios
        if n_vazios > 0:
            log(f"[dedup] {n_vazios} chunk(s) sem conteúdo descartado(s).")
        if n_duplicados > 0:
            log(f"[dedup] {n_duplicados} chunks duplicados removidos. "
                f"Restam {len(chunks_unicos)} únicos.")
        chunks = chunks_unicos
        tempo_chunk = time.perf_counter() - t_chunk_ini
        log(f"[tempo] Chunking + deduplicação: {self._fmt_tempo(tempo_chunk)}.")

        log("[index] Indexando no vector store (ChromaDB)...")
        t_index_ini = time.perf_counter()
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            collection_name=collection,
            persist_directory=persist_dir,
        )
        tempo_index = time.perf_counter() - t_index_ini
        tempo_total = time.perf_counter() - t_inicio

        chunks_por_s = round(len(chunks) / tempo_index, 1) if tempo_index else 0.0

        log(f"[ok] {len(chunks)} chunks indexados.")
        log(f"[tempo] Indexação vetorial (embeddings + ChromaDB): "
            f"{self._fmt_tempo(tempo_index)}.")
        log(
            f"[tempo] TOTAL da indexação: {self._fmt_tempo(tempo_total)} "
            f"(carga {tempo_carga:.1f}s + chunking {tempo_chunk:.1f}s + "
            f"indexação {tempo_index:.1f}s) para {len(chunks)} chunks "
            f"[{chunks_por_s:.0f} chunks/s na indexação]."
        )

        self.tempos_indexacao = {
            "reindexado": True,
            "total_documentos": len(documentos),
            "total_chunks": len(chunks),
            "chunks_duplicados_removidos": n_duplicados,
            "tempo_carga_documentos_s": round(tempo_carga, 2),
            "tempo_chunking_s": round(tempo_chunk, 2),
            "tempo_indexacao_vetorial_s": round(tempo_index, 2),
            "tempo_total_indexacao_s": round(tempo_total, 2),
            "tempo_total_indexacao_min": round(tempo_total / 60, 2),
            "chunks_por_segundo_indexacao": chunks_por_s,
        }
        return len(chunks)

    def _get_retriever(self):
        if self._vector_store is None:
            raise RuntimeError("Base não indexada. Chame indexar_base() antes.")
        top_k = self.params.get("top_k", 8)
        search_type = self.params.get("search_type", "mmr")

        if search_type == "mmr":
            # MMR (Maximal Marginal Relevance): busca diversidade nos resultados,
            # evitando chunks redundantes/duplicados nos top-k.
            return self._vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": top_k * 3,  # busca 3x mais candidatos para diversificar
                    "lambda_mult": self.params.get("mmr_lambda", 0.7),
                },
            )
        # Fallback: similaridade pura
        return self._vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

    def _recuperar_trechos(self, pergunta: str) -> list[dict[str, Any]]:
        """
        Recupera os trechos relevantes buscando DIRETO na collection do Chroma.

        Isso evita que o LangChain instancie objetos ``Document`` a partir de
        registros com texto nulo (o que quebraria a validação do pydantic).
        Aplica MMR manualmente quando ``search_type='mmr'`` e descarta trechos
        sem conteúdo.

        Retorna uma lista de dicts: {"conteudo": str, "metadata": dict}.
        """
        if self._vector_store is None:
            raise RuntimeError("Base não indexada. Chame indexar_base() antes.")

        import numpy as np

        top_k = self.params.get("top_k", 8)
        search_type = self.params.get("search_type", "mmr")
        collection = self._vector_store._collection

        # Embedding da pergunta (mesma função usada na indexação).
        emb_query = self._get_embeddings().embed_query(pergunta)

        if search_type == "mmr":
            fetch_k = top_k * 3
            lambda_mult = self.params.get("mmr_lambda", 0.7)
            res = collection.query(
                query_embeddings=[emb_query],
                n_results=fetch_k,
                include=["documents", "metadatas", "embeddings"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            embs = (res.get("embeddings") or [[]])[0]

            # Filtra candidatos com conteúdo válido.
            candidatos = [
                (doc, meta, emb)
                for doc, meta, emb in zip(docs, metas, embs)
                if doc is not None and str(doc).strip()
            ]
            if not candidatos:
                return []

            # MMR manual: seleciona iterativamente equilibrando relevância
            # (similaridade com a query) e diversidade (dissimilaridade com os
            # já escolhidos).
            q = np.array(emb_query, dtype=float)

            def _cos(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

            cand_embs = [np.array(e, dtype=float) for _, _, e in candidatos]
            sim_query = [_cos(q, e) for e in cand_embs]

            selecionados: list[int] = []
            restantes = list(range(len(candidatos)))
            while restantes and len(selecionados) < top_k:
                if not selecionados:
                    melhor = max(restantes, key=lambda i: sim_query[i])
                else:
                    def score(i):
                        div = max(
                            _cos(cand_embs[i], cand_embs[j]) for j in selecionados
                        )
                        return lambda_mult * sim_query[i] - (1 - lambda_mult) * div
                    melhor = max(restantes, key=score)
                selecionados.append(melhor)
                restantes.remove(melhor)

            return [
                {"conteudo": str(candidatos[i][0]), "metadata": candidatos[i][1] or {}}
                for i in selecionados
            ]

        # Similaridade pura
        res = collection.query(
            query_embeddings=[emb_query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        return [
            {"conteudo": str(doc), "metadata": meta or {}}
            for doc, meta in zip(docs, metas)
            if doc is not None and str(doc).strip()
        ]

    # -- Chamadas às LLMs (com cronometragem) --------------------------------

    def _invoke_llm(self, prompt: str, on_wait: Callable | None = None) -> str:
        """Invoca a LLM medindo apenas o tempo de geração (sem backoff)."""
        from langchain_core.messages import HumanMessage

        llm = self._get_llm()

        def chamada():
            inicio = time.perf_counter()
            resposta = llm.invoke([HumanMessage(content=prompt)])
            decorrido = time.perf_counter() - inicio
            return resposta.content, decorrido

        # com_backoff repete a chamada em caso de 429; cada tentativa
        # cronometra somente a execução real da LLM.
        return com_backoff(
            chamada,
            max_retries=self.rate.get("max_retries", 6),
            base_delay_s=self.rate.get("base_delay_s", 2.0),
            max_delay_s=self.rate.get("max_delay_s", 60.0),
            on_wait=on_wait,
        )

    def _formatar_prompt(self, template: str, **kwargs) -> str:
        """
        Formata o template preenchendo o domínio do tema e demais campos.

        Tolerante a placeholders ausentes: se um template não usar {dominio}
        (ex.: configs antigas), a formatação ainda funciona.
        """
        campos = {"dominio": self.dominio, **kwargs}
        try:
            return template.format(**campos)
        except KeyError:
            # Placeholder desconhecido no template: aplica só os campos válidos.
            for chave, valor in campos.items():
                template = template.replace("{" + chave + "}", str(valor))
            return template

    def executar_padrao(self, pergunta: str, on_wait=None) -> dict[str, Any]:
        """LLM Padrão: pergunta direta, sem contexto."""
        prompt = self._formatar_prompt(self.prompts["llm_padrao"], pergunta=pergunta)
        resposta, tempo = self._invoke_llm(prompt, on_wait=on_wait)
        return {"resposta": resposta, "tempo_execucao_padrao_s": round(tempo, 4)}

    def executar_rag(self, pergunta: str, on_wait=None) -> dict[str, Any]:
        """LLM com RAG: retrieval + geração, cronometrados separadamente."""
        # 1) Retrieval (embedding da pergunta + busca vetorial).
        # Busca direto na collection do Chroma (dicts crus), evitando que o
        # LangChain instancie Documents a partir de registros com texto nulo.
        t0 = time.perf_counter()
        trechos = self._recuperar_trechos(pergunta)
        tempo_retrieval = time.perf_counter() - t0

        contexto = "\n\n---\n\n".join(t["conteudo"] for t in trechos)
        fontes = [
            {"conteudo": t["conteudo"][:500], "metadata": t["metadata"]}
            for t in trechos
        ]

        # 2) Geração com contexto
        prompt = self._formatar_prompt(
            self.prompts["llm_rag"],
            contexto_recuperado=contexto,
            pergunta=pergunta,
        )
        resposta, tempo_geracao = self._invoke_llm(prompt, on_wait=on_wait)

        return {
            "resposta": resposta,
            "contexto_recuperado": contexto,
            "fontes": fontes,
            "tempo_retrieval_rag_s": round(tempo_retrieval, 4),
            "tempo_geracao_rag_s": round(tempo_geracao, 4),
            "tempo_total_rag_s": round(tempo_retrieval + tempo_geracao, 4),
        }


# ---------------------------------------------------------------------------
# Orquestração de alto nível (usada por CLI e UI)
# ---------------------------------------------------------------------------

def executar_benchmarking(
    config: Config,
    perguntas: list[dict],
    gabarito: list[dict],
    modelo_execucao: str | None = None,
    modelo_juiz: str | None = None,
    log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, dict], None] | None = None,
    reindexar: bool = False,
) -> dict[str, Any]:
    """
    Executa o benchmarking completo para todas as perguntas.

    Args:
        config: configuração carregada.
        perguntas: lista de perguntas [{id, pergunta, ...}].
        gabarito: lista do gabarito [{id, gabarito_oficial, ...}].
        modelo_execucao / modelo_juiz: overrides opcionais de modelo.
        log: callback para mensagens de log (console/UI).
        on_progress: callback (indice, total, resultado_parcial) por pergunta.
        reindexar: força reindexação da base vetorial.

    Returns:
        Relatório no formato do contrato de saída.
    """
    from .judge import Judge
    from .datajud import validar_processos_citados
    from .utils import (
        indexar_por_id,
        montar_relatorio,
        calcular_metricas_globais,
    )

    log = log or (lambda m: None)

    pipeline = Pipeline(config, modelo_execucao=modelo_execucao)
    juiz = Judge(config, modelo_juiz=modelo_juiz)

    validar_datajud = config.parametros.get("validar_processos_datajud", True)
    if validar_datajud:
        log("[datajud] Validação de processos citados: ATIVADA (TRFs).")
    else:
        log("[datajud] Validação de processos citados: DESATIVADA.")

    log("[rag] Preparando base de conhecimento (RAG)...")
    pipeline.indexar_base(config.dados["base_juridica"], log=log, forcar=reindexar)

    gab_idx = indexar_por_id(gabarito)
    total = len(perguntas)
    resultados: list[dict] = []

    for i, p in enumerate(perguntas, start=1):
        pid = str(p.get("id", p.get("id_pergunta", i)))
        pergunta = p.get("pergunta", "")
        gab_item = gab_idx.get(pid, {})
        gab_texto = gab_item.get("gabarito_oficial", gab_item.get("resposta_referencia", ""))

        log(f"\n> Pergunta {i}/{total} ({pid}): {pergunta[:60]}...")

        def _wait_log(espera, tentativa):
            log(f"   [429] Rate limit. Backoff {espera:.1f}s (tentativa {tentativa})...")

        # LLM Padrão
        r_padrao = pipeline.executar_padrao(pergunta, on_wait=_wait_log)
        log(f"   [padrao] LLM Padrão respondeu em {r_padrao['tempo_execucao_padrao_s']}s")

        # LLM RAG
        r_rag = pipeline.executar_rag(pergunta, on_wait=_wait_log)
        log(
            f"   [rag] LLM RAG respondeu em {r_rag['tempo_total_rag_s']}s "
            f"(retrieval {r_rag['tempo_retrieval_rag_s']}s + "
            f"geração {r_rag['tempo_geracao_rag_s']}s)"
        )

        # Juiz
        aval = juiz.avaliar(
            pergunta=pergunta,
            gabarito=gab_texto,
            resposta_padrao=r_padrao["resposta"],
            resposta_rag=r_rag["resposta"],
            on_wait=_wait_log,
        )
        log(f"   [juiz] Juiz avaliou em {aval['tempo_avaliacao_juiz_s']}s")

        def _fmt_notas(rotulo: str, av: dict):
            aluc = "SIM" if av.get("alucinacao_detectada") else "não"
            fund = "SIM" if av.get("fundamentacao_correta") else "não"
            log(
                f"   [juiz:{rotulo}] fidelidade={av.get('nota_fidelidade')}/5 | "
                f"precisão={av.get('nota_precisao_gabarito')}/5 | "
                f"completude={av.get('nota_completude')}/5 | "
                f"alucinação={aluc} | fundamentação={fund}"
            )
            justificativa = av.get("justificativa_juiz")
            if justificativa:
                log(f"   [juiz:{rotulo}] justificativa: {justificativa}")

        _fmt_notas("padrao", aval["avaliacao_llm_padrao"])
        _fmt_notas("rag", aval["avaliacao_llm_rag"])

        # Validação factual de processos citados (DataJud - TRFs), se habilitada
        if validar_datajud:
            palavras_tema = config.tema.get("palavras_chave_datajud")
            validacao_padrao = validar_processos_citados(
                r_padrao["resposta"], palavras_tema=palavras_tema
            )
            validacao_rag = validar_processos_citados(
                r_rag["resposta"], palavras_tema=palavras_tema
            )
        else:
            validacao_padrao = []
            validacao_rag = []

        def _log_validacao(rotulo: str, procs: list[dict]):
            if not procs:
                return
            log(f"   [datajud:{rotulo}] {len(procs)} processo(s) citado(s):")
            for p in procs:
                if p.get("erro"):
                    log(f"      - {p['numero']} -> {p['erro']}")
                elif not p.get("existe"):
                    log(f"      - {p['numero']} -> NÃO ENCONTRADO no DataJud (possível alucinação)")
                else:
                    compat = "compatível" if p.get("compativel_tema") else "INCOMPATÍVEL"
                    assuntos = ", ".join(p.get("assuntos", [])) or "(sem assunto)"
                    log(
                        f"      - {p['numero']} -> existe | assunto: {assuntos} "
                        f"| tema: {compat}"
                    )

        _log_validacao("padrao", validacao_padrao)
        _log_validacao("rag", validacao_rag)

        resultado = {
            "id_pergunta": pid,
            "pergunta": pergunta,
            "gabarito_oficial": gab_texto,
            "resposta_llm_padrao": r_padrao["resposta"],
            "tempo_execucao_padrao_s": r_padrao["tempo_execucao_padrao_s"],
            "resposta_llm_rag": r_rag["resposta"],
            "tempo_retrieval_rag_s": r_rag["tempo_retrieval_rag_s"],
            "tempo_geracao_rag_s": r_rag["tempo_geracao_rag_s"],
            "tempo_total_rag_s": r_rag["tempo_total_rag_s"],
            "fontes_rag": r_rag["fontes"],
            "avaliacao_llm_padrao": aval["avaliacao_llm_padrao"],
            "avaliacao_llm_rag": aval["avaliacao_llm_rag"],
            "tempo_avaliacao_juiz_s": aval["tempo_avaliacao_juiz_s"],
            "validacao_processos_padrao": validacao_padrao,
            "validacao_processos_rag": validacao_rag,
        }
        resultados.append(resultado)

        if on_progress:
            on_progress(i, total, resultado)

    # Parecer final consolidado da LLM Juiz: analisa todas as avaliações e
    # posiciona qual abordagem (Padrão vs RAG) teve melhor desempenho.
    parecer = {}
    if resultados:
        log("\n[juiz] Gerando parecer final consolidado...")

        def _wait_parecer(espera, tentativa):
            log(f"   [429] Rate limit. Backoff {espera:.1f}s (tentativa {tentativa})...")

        try:
            metricas = calcular_metricas_globais(resultados)
            parecer = juiz.parecer_final(
                metricas, resultados, on_wait=_wait_parecer
            )
            log(f"[juiz] Abordagem vencedora: {parecer.get('abordagem_vencedora')}")
            log("[juiz] Parecer final:")
            for linha in parecer.get("parecer_texto", "").splitlines():
                log(f"   {linha}")
        except Exception as exc:  # noqa: BLE001
            log(f"[juiz] Falha ao gerar parecer final: {exc}")
            parecer = {}

    return montar_relatorio(
        resultados,
        modelo_testado=modelo_execucao or config.modelos["llm_execucao"],
        modelo_juiz=modelo_juiz or config.modelos["llm_juiz"],
        tempos_indexacao=pipeline.tempos_indexacao,
        parecer_final=parecer,
    )
