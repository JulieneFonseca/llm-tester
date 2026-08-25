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

    def indexar_base(
        self,
        diretorio: str,
        log: Callable[[str], None] | None = None,
        forcar: bool = False,
    ) -> int:
        """
        Carrega documentos, aplica chunking (1000/150) e indexa no ChromaDB.
        Retorna o número de chunks indexados.
        """
        log = log or (lambda m: None)
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        persist_dir = self.params.get("persist_directory", "data/chroma_db")
        collection = self.params.get("collection_name", "pensao_por_morte")

        if not forcar and Path(persist_dir).exists():
            log("[base] Vector store existente encontrado. Carregando...")
            self._vector_store = Chroma(
                collection_name=collection,
                embedding_function=self._get_embeddings(),
                persist_directory=persist_dir,
            )
            return self._vector_store._collection.count()

        base = Path(diretorio)
        if not base.exists():
            raise FileNotFoundError(f"Base jurídica não encontrada: {diretorio}")

        documentos = []
        arquivos = list(base.glob("**/*.pdf")) + list(base.glob("**/*.txt"))
        log(f"[docs] {len(arquivos)} arquivo(s) na base jurídica.")

        for arq in arquivos:
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

        if not documentos:
            raise ValueError(
                f"Nenhum documento carregado de '{diretorio}'. "
                "Adicione PDFs ou TXTs à base jurídica."
            )

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

        log("[index] Indexando no vector store (ChromaDB)...")
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            collection_name=collection,
            persist_directory=persist_dir,
        )
        log(f"[ok] {len(chunks)} chunks indexados.")
        return len(chunks)

    def _get_retriever(self):
        if self._vector_store is None:
            raise RuntimeError("Base não indexada. Chame indexar_base() antes.")
        return self._vector_store.as_retriever(
            search_kwargs={"k": self.params.get("top_k", 4)}
        )

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
        retriever = self._get_retriever()

        # 1) Retrieval (embedding da pergunta + busca vetorial)
        t0 = time.perf_counter()
        docs = retriever.invoke(pergunta)
        tempo_retrieval = time.perf_counter() - t0

        contexto = "\n\n---\n\n".join(d.page_content for d in docs)
        fontes = [
            {"conteudo": d.page_content[:500], "metadata": d.metadata}
            for d in docs
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
    from .utils import indexar_por_id, montar_relatorio

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
                    log(f"      • {p['numero']} → {p['erro']}")
                elif not p.get("existe"):
                    log(f"      • {p['numero']} → NÃO ENCONTRADO no DataJud (possível alucinação)")
                else:
                    compat = "compatível" if p.get("compativel_tema") else "INCOMPATÍVEL"
                    assuntos = ", ".join(p.get("assuntos", [])) or "(sem assunto)"
                    log(
                        f"      • {p['numero']} → existe | assunto: {assuntos} "
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

    return montar_relatorio(
        resultados,
        modelo_testado=modelo_execucao or config.modelos["llm_execucao"],
        modelo_juiz=modelo_juiz or config.modelos["llm_juiz"],
    )
