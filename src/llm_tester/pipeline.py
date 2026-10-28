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
        # Modo "conteúdo explícito": quando `campos_conteudo` é definido, apenas
        # essas colunas formam o texto do chunk (SEM truncar) e `campos_metadados`
        # vira metadata. Ideal para jurisprudência (ex.: a EMENTA é o resumo
        # oficial do acórdão e deve ir inteira para o índice, enquanto a DECISÃO
        # COMPLETA fica fora). Se vazio, usa a lógica genérica antiga.
        campos_conteudo = csv_config.get("campos_conteudo") or []
        campos_metadados = csv_config.get("campos_metadados") or []
        limite_metadado = csv_config.get("limite_metadado_chars", 200)
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

        import re

        def _norm(nome: str) -> str:
            """Normaliza nome de coluna p/ casar sem depender de caixa/espaços."""
            return re.sub(r"\s+", " ", str(nome).strip()).casefold()

        conteudo_norm = {_norm(c) for c in campos_conteudo}
        metadados_norm = {_norm(c) for c in campos_metadados}
        ignorar_norm = {_norm(c) for c in campos_ignorar}
        modo_explicito = bool(conteudo_norm)

        for idx, row in enumerate(linhas):
            partes_conteudo: list[str] = []
            metadata: dict[str, Any] = {"source": str(caminho), "csv_row": idx}

            def _valor_str(valor) -> str:
                if isinstance(valor, (list, tuple)):
                    return " ".join(str(v) for v in valor if v is not None).strip()
                return str(valor).strip() if valor is not None else ""

            if modo_explicito:
                # MODO 1 (jurisprudência): só os campos escolhidos viram conteúdo,
                # SEM truncar; os demais escolhidos viram metadata; resto ignorado.
                #
                # IMPORTANTE (precisão do retrieval): o TEXTO do chunk (que vira o
                # embedding) contém APENAS a ementa — o cabeçalho administrativo
                # (arquivo/processo/tribunal) NÃO entra no vetor, pois é igual em
                # todas as decisões e "aproximaria" ementas sem relação, piorando
                # a busca semântica. A origem fica em METADADOS e é reconstruída
                # no contexto do RAG (executar_rag) a partir deles, para a LLM
                # ainda poder citar a decisão e o arquivo de onde ela veio.
                import os as _os
                valores_norm = {_norm(c): _valor_str(v) for c, v in row.items()
                                if c is not None}

                def _primeiro(*nomes: str) -> str:
                    for n in nomes:
                        v = valores_norm.get(_norm(n))
                        if v:
                            return v
                    return ""

                arquivo_origem = _os.path.basename(str(caminho))
                num_processo = _primeiro("PROCESSO JUDICIAL", "Nº", "processo")
                tribunal = _primeiro("TRIBUNAL")
                # Guarda a origem em metadados (NÃO no texto vetorizado).
                metadata["arquivo_origem"] = arquivo_origem
                if num_processo:
                    metadata["processo_acordao"] = num_processo[:limite_metadado]
                if tribunal:
                    metadata["tribunal_origem"] = tribunal[:limite_metadado]

                for campo, valor in row.items():
                    if campo is None:
                        continue
                    cn = _norm(campo)
                    valor_limpo = _valor_str(valor)
                    if not valor_limpo or cn in ignorar_norm:
                        continue
                    if cn in conteudo_norm:
                        partes_conteudo.append(f"{campo}: {valor_limpo}")
                    elif cn in metadados_norm:
                        # Metadado curto (evita estourar o tamanho do metadata).
                        metadata[campo] = valor_limpo[:limite_metadado]
            else:
                # MODO LEGADO: concatena tudo, resumindo campos grandes.
                for campo, valor in row.items():
                    if campo is None:
                        continue
                    if valor is None or _norm(campo) in ignorar_norm:
                        continue
                    valor_limpo = _valor_str(valor)
                    if not valor_limpo:
                        continue
                    if len(valor_limpo) > limite_campo:
                        valor_resumido = self._resumir_campo(valor_limpo, resumo_max)
                        partes_conteudo.append(f"{campo}: {valor_resumido}")
                        campos_resumidos_log.add(campo)
                    else:
                        partes_conteudo.append(f"{campo}: {valor_limpo}")
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

    # -- Helpers de indexação (reutilizados por indexar_base e indexar_arquivo) --

    def _chroma_settings(self):
        """
        Settings do ChromaDB com persistência garantida e telemetria desligada.

        is_persistent + persist_directory são ESSENCIAIS: sem eles o
        langchain-chroma pode criar um cliente EM MEMÓRIA e os dados somem ao
        encerrar o processo (foi a causa do banco ficar vazio após "indexar").
        """
        from chromadb.config import Settings

        persist_dir = self.params.get("persist_directory", "data/chroma_db")
        return Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=persist_dir,
        )

    def _abrir_vector_store(self):
        """Abre/cria a collection persistente e guarda em self._vector_store."""
        from langchain_community.vectorstores import Chroma

        persist_dir = self.params.get("persist_directory", "data/chroma_db")
        collection = self.params.get("collection_name", "pensao_por_morte")
        self._vector_store = Chroma(
            collection_name=collection,
            embedding_function=self._get_embeddings(),
            persist_directory=persist_dir,
            client_settings=self._chroma_settings(),
        )
        return self._vector_store

    def _apagar_collection(self, log) -> int:
        """
        Apaga a collection inteira do ChromaDB (zera a base), de forma confiável.

        Deletar por IDs é frágil com dezenas de milhares de chunks (col.get()
        sem paginação não retorna tudo). Apagar a collection e recriá-la é o
        único jeito seguro. Retorna quantos chunks existiam antes.
        """
        persist_dir = self.params.get("persist_directory", "data/chroma_db")
        collection = self.params.get("collection_name", "pensao_por_morte")
        antes = 0
        if Path(persist_dir).exists():
            try:
                import chromadb
                client = chromadb.PersistentClient(
                    path=persist_dir, settings=self._chroma_settings()
                )
                try:
                    antes = client.get_collection(collection).count()
                except Exception:
                    antes = 0
                client.delete_collection(collection)
                log(f"[base] Collection '{collection}' apagada ({antes} chunk(s)).")
            except Exception as exc:  # noqa: BLE001
                log(f"[aviso] Falha ao apagar collection (pode não existir): {exc}")
        # (Re)cria a collection vazia.
        self._abrir_vector_store()
        return antes

    def _carregar_arquivos(self, arquivos_doc, arquivos_csv, log) -> list:
        """Carrega uma lista de arquivos (PDF/TXT/CSV) em Documents."""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        documentos = []
        for arq in arquivos_doc:
            try:
                if arq.suffix.lower() == ".pdf":
                    documentos.extend(PyPDFLoader(str(arq)).load())
                else:
                    documentos.extend(TextLoader(str(arq), encoding="utf-8").load())
                log(f"   [ok] {arq.name}")
            except Exception as exc:  # noqa: BLE001
                log(f"   [erro] Falha ao ler {arq.name}: {exc}")

        for arq in arquivos_csv:
            try:
                docs_csv = self._carregar_csv_base(arq, log)
                documentos.extend(docs_csv)
                log(f"   [ok] {arq.name} ({len(docs_csv)} registros)")
            except Exception as exc:  # noqa: BLE001
                log(f"   [erro] Falha ao ler {arq.name}: {exc}")

        return documentos

    def _chunkar_documentos(self, documentos: list, log) -> tuple[list, int]:
        """Aplica chunking + sanitização + deduplicação. Retorna (chunks, n_dup)."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        return chunks_unicos, n_duplicados

    def _inserir_chunks_lotes(self, chunks: list, log) -> float:
        """
        Insere os chunks no vector store em LOTES (gravação incremental),
        emitindo progresso com ETA. Retorna o tempo total de indexação (s).

        Motivo dos lotes: gerar embeddings de dezenas de milhares de chunks de
        uma vez estoura a RAM. Em blocos de `index_batch_size`, a memória é
        liberada entre um lote e outro.
        """
        import datetime as _dt

        batch_size = int(self.params.get("index_batch_size", 500))
        total_chunks = len(chunks)
        log(f"[index] Indexando {total_chunks} chunks no ChromaDB "
            f"em lotes de {batch_size}...")

        t_index_ini = time.perf_counter()
        n_lotes = (total_chunks + batch_size - 1) // batch_size
        for i_lote, inicio in enumerate(range(0, total_chunks, batch_size), start=1):
            lote = chunks[inicio : inicio + batch_size]

            t_lote_ini = time.perf_counter()
            self._vector_store.add_documents(lote)
            dt_lote = time.perf_counter() - t_lote_ini

            processados = min(inicio + batch_size, total_chunks)
            pct = (processados / total_chunks) * 100 if total_chunks else 100.0
            parcial = time.perf_counter() - t_index_ini
            vel = processados / parcial if parcial > 0 else 0.0
            restantes = total_chunks - processados
            eta_s = restantes / vel if vel > 0 else 0.0
            agora = _dt.datetime.now().strftime("%H:%M:%S")

            log(
                f"   [index {agora}] lote {i_lote}/{n_lotes} | "
                f"{processados}/{total_chunks} ({pct:.0f}%) | "
                f"lote={dt_lote:.1f}s | {vel:.0f} chunks/s | "
                f"decorrido={self._fmt_tempo(parcial)} | "
                f"ETA={self._fmt_tempo(eta_s)}"
            )

        return time.perf_counter() - t_index_ini

    def _remover_source(self, caminho: Path, log) -> int:
        """
        Remove do ChromaDB todos os chunks cujo metadado `source` corresponde
        ao arquivo dado. Usado antes de reindexar um arquivo, para evitar
        duplicatas. Retorna quantos chunks foram removidos.

        Eficiência: a deleção é feita por FILTRO (`where`) direto no Chroma, que
        resolve no banco sem carregar todos os metadados na RAM (evita o erro
        'Memória insuficiente' em bases grandes). O `source` gravado pode estar
        em caminho relativo OU absoluto; por isso tentamos ambas as variações.
        """
        col = self._vector_store._collection
        removidos = 0

        # Variações de caminho que podem ter sido gravadas em `source`.
        candidatos = {
            str(caminho),
            str(caminho.resolve()),
            caminho.as_posix(),
            caminho.resolve().as_posix(),
        }

        for src in candidatos:
            try:
                # delete(where=...) deixa o Chroma filtrar e apagar no banco,
                # sem materializar os metadados de toda a coleção em memória.
                existentes = col.get(where={"source": src}, include=[])
                ids = existentes.get("ids", []) or []
                if ids:
                    col.delete(ids=ids)
                    removidos += len(ids)
            except Exception as exc:  # noqa: BLE001
                log(f"   [aviso] Falha ao remover chunks de {src}: {exc}")

        if removidos:
            log(f"[limpeza] {removidos} chunk(s) antigos de '{caminho.name}' removidos.")
        else:
            log(f"[limpeza] Nenhum chunk antigo de '{caminho.name}' encontrado "
                f"para remover (ok se for a 1ª indexação do arquivo).")
        return removidos

    # -- Indexação completa --------------------------------------------------

    def indexar_base(
        self,
        diretorio: str,
        log: Callable[[str], None] | None = None,
        forcar: bool = False,
    ) -> int:
        """
        Carrega documentos (PDF, TXT, CSV), aplica chunking e indexa no ChromaDB.
        Retorna o número de chunks indexados.

        - forcar=False: se já existe base indexada, apenas carrega (não reindexa).
        - forcar=True: reprocessa TODA a base do zero.
        """
        log = log or (lambda m: None)

        persist_dir = self.params.get("persist_directory", "data/chroma_db")

        if not forcar and Path(persist_dir).exists():
            t0 = time.perf_counter()
            log("[base] Vetor existente encontrado. Carregando...")
            self._abrir_vector_store()
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

        arquivos_doc = list(base.glob("**/*.pdf")) + list(base.glob("**/*.txt"))
        arquivos_csv = list(base.glob("**/*.csv"))
        total_arquivos = len(arquivos_doc) + len(arquivos_csv)
        log(f"[docs] {total_arquivos} arquivo(s) na base jurídica "
            f"({len(arquivos_doc)} PDF/TXT + {len(arquivos_csv)} CSV).")

        # Reindexação total: começa do zero, APAGANDO a collection inteira.
        log("[base] Reindexação total: apagando collection existente...")
        self._apagar_collection(log)

        t_carga_ini = time.perf_counter()
        documentos = self._carregar_arquivos(arquivos_doc, arquivos_csv, log)
        if not documentos:
            raise ValueError(
                f"Nenhum documento carregado de '{diretorio}'. "
                "Adicione PDFs, TXTs ou CSVs à base jurídica."
            )
        tempo_carga = time.perf_counter() - t_carga_ini
        log(f"[tempo] Carga dos documentos: {self._fmt_tempo(tempo_carga)} "
            f"({len(documentos)} documentos).")

        t_chunk_ini = time.perf_counter()
        chunks, n_duplicados = self._chunkar_documentos(documentos, log)
        tempo_chunk = time.perf_counter() - t_chunk_ini
        log(f"[tempo] Chunking + deduplicação: {self._fmt_tempo(tempo_chunk)}.")

        tempo_index = self._inserir_chunks_lotes(chunks, log)
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

    # -- Indexação incremental (por arquivo) ---------------------------------

    def _indexar_um_arquivo(self, arq: Path, log, remover_antigo: bool = True) -> int:
        """
        Indexa um único arquivo na collection JÁ ABERTA (self._vector_store).
        Se `remover_antigo`, remove os chunks anteriores desse `source` antes
        (evita duplicatas em reindexações incrementais). Retorna nº de chunks.
        Reutilizado por indexar_arquivo e indexar_selecionados.
        """
        if remover_antigo:
            self._remover_source(arq, log)

        suf = arq.suffix.lower()
        if suf == ".csv":
            arquivos_doc, arquivos_csv = [], [arq]
        elif suf in (".pdf", ".txt"):
            arquivos_doc, arquivos_csv = [arq], []
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {suf}")

        documentos = self._carregar_arquivos(arquivos_doc, arquivos_csv, log)
        if not documentos:
            log(f"[aviso] Nenhum conteúdo carregado de '{arq.name}'.")
            return 0

        chunks, _ = self._chunkar_documentos(documentos, log)
        self._inserir_chunks_lotes(chunks, log)
        return len(chunks)

    def indexar_arquivo(
        self,
        caminho: str,
        log: Callable[[str], None] | None = None,
    ) -> int:
        """
        Indexa (ou reindexa) UM único arquivo, sem afetar o resto da base.
        Remove os chunks antigos daquele arquivo antes (evita duplicatas).
        Retorna o número de chunks inseridos para o arquivo.
        """
        log = log or (lambda m: None)

        arq = Path(caminho)
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        t_inicio = time.perf_counter()
        log(f"[arquivo] Indexação incremental de: {arq.name}")

        self._abrir_vector_store()
        n = self._indexar_um_arquivo(arq, log, remover_antigo=True)
        tempo_total = time.perf_counter() - t_inicio

        total_atual = self._vector_store._collection.count()
        log(f"[ok] {n} chunks de '{arq.name}' indexados em {self._fmt_tempo(tempo_total)}.")
        log(f"[base] Total de chunks na base agora: {total_atual}")

        self.tempos_indexacao = {
            "reindexado": True,
            "arquivo": arq.name,
            "total_chunks": n,
            "total_chunks_base": total_atual,
            "tempo_total_indexacao_s": round(tempo_total, 2),
        }
        return n

    # -- Indexação de uma seleção de arquivos --------------------------------

    def indexar_selecionados(
        self,
        caminhos: list[str],
        log: Callable[[str], None] | None = None,
        zerar: bool = False,
    ) -> int:
        """
        Indexa uma LISTA de arquivos selecionados.

        - zerar=False: incremental. Atualiza cada arquivo da lista (removendo a
          versão antiga de cada um), mantendo o resto da base intacto.
        - zerar=True: apaga TODA a base primeiro e reindexa apenas os arquivos
          da lista.

        Se a lista estiver vazia, NÃO faz nada (nem zera). Retorna o total de
        chunks inseridos nesta operação.
        """
        log = log or (lambda m: None)

        arquivos = [Path(c) for c in caminhos]
        existentes = [a for a in arquivos if a.exists()]
        faltando = [a for a in arquivos if not a.exists()]
        for a in faltando:
            log(f"[aviso] Arquivo não encontrado, ignorado: {a}")

        if not existentes:
            log("[base] Nenhum arquivo selecionado válido. Nada a indexar.")
            self.tempos_indexacao = {"reindexado": False, "total_chunks": 0}
            return 0

        t_inicio = time.perf_counter()
        modo = "ZERAR + reindexar selecionados" if zerar else "incremental (selecionados)"
        log(f"[base] Indexação de {len(existentes)} arquivo(s) — modo: {modo}.")

        if zerar:
            log("[base] Apagando a base antes de reindexar os selecionados...")
            self._apagar_collection(log)
            remover_antigo = False  # base já vazia, não precisa remover por source
        else:
            self._abrir_vector_store()
            remover_antigo = True

        total = 0
        for i, arq in enumerate(existentes, start=1):
            log(f"\n[{i}/{len(existentes)}] Indexando: {arq.name}")
            total += self._indexar_um_arquivo(arq, log, remover_antigo=remover_antigo)

        tempo_total = time.perf_counter() - t_inicio
        total_atual = self._vector_store._collection.count()
        log(f"\n[ok] {total} chunks inseridos ({len(existentes)} arquivo(s)) "
            f"em {self._fmt_tempo(tempo_total)}.")
        log(f"[base] Total de chunks na base agora: {total_atual}")

        self.tempos_indexacao = {
            "reindexado": True,
            "zerou_base": zerar,
            "arquivos": [a.name for a in existentes],
            "total_chunks": total,
            "total_chunks_base": total_atual,
            "tempo_total_indexacao_s": round(tempo_total, 2),
        }
        return total

    # -- Priorização de jurisprudência (arquivos CJF) ------------------------
    #
    # Termos que indicam que a pergunta pede DECISÕES/JULGADOS (jurisprudência),
    # e não apenas a fundamentação legal. Quando detectados, o retrieval reserva
    # vagas no top-k para chunks vindos dos arquivos CJF (jurisprudência), para
    # que os PDFs de legislação não ocupem todas as posições.
    _TERMOS_PEDE_JURISPRUDENCIA = (
        "decis", "julgad", "acórdão", "acordao", "acórdãos", "acordaos",
        "jurisprud", "precedente", "ementa", "provas judiciais", "prova judicial",
        "provas judicias", "comprov",  # "comprovação" costuma pedir julgados no tema
    )

    @staticmethod
    def _pede_jurisprudencia(pergunta: str) -> bool:
        """True se a pergunta pede decisões/julgados/jurisprudência."""
        t = (pergunta or "").lower()
        return any(k in t for k in Pipeline._TERMOS_PEDE_JURISPRUDENCIA)

    @staticmethod
    def _num_julgados_solicitados(pergunta: str) -> int:
        """
        Extrai a quantidade de julgados pedida na pergunta (ex.: '3 decisões',
        'traga 5 julgados'). Retorna 0 se não houver número explícito.
        """
        import re as _re
        t = (pergunta or "").lower()
        # Número seguido (em até ~3 palavras) de um termo de julgado/decisão.
        m = _re.search(
            r"(\d{1,2})\s+(?:\w+\s+){0,3}?(decis|julgad|acórdão|acordao|precedente|ementa)",
            t,
        )
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _e_chunk_cjf(meta: dict | None) -> bool:
        """True se o chunk veio de um arquivo cujo nome começa com 'CJF'."""
        import os as _os
        if not meta:
            return False
        origem = meta.get("source") or meta.get("fonte") or ""
        nome = _os.path.basename(str(origem))
        return nome.upper().startswith("CJF")

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
        q = np.array(emb_query, dtype=float)

        def _cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

        def _mmr(candidatos: list[tuple], limite: int,
                 forcar_relevancia: bool = False) -> list[tuple]:
            """
            Aplica MMR (relevância × diversidade) sobre `candidatos`
            [(doc, meta, emb), ...], retornando até `limite` itens.
            Se search_type != 'mmr' OU forcar_relevancia=True, ordena só por
            similaridade com a query (os mais aderentes primeiro, sem penalizar
            por diversidade).
            """
            if not candidatos or limite <= 0:
                return []
            cand_embs = [np.array(e, dtype=float) for _, _, e in candidatos]
            sim_query = [_cos(q, e) for e in cand_embs]
            if search_type != "mmr" or forcar_relevancia:
                ordem = sorted(range(len(candidatos)),
                               key=lambda i: sim_query[i], reverse=True)
                return [candidatos[i] for i in ordem[:limite]]
            lambda_mult = self.params.get("mmr_lambda", 0.7)
            selecionados: list[int] = []
            restantes = list(range(len(candidatos)))
            while restantes and len(selecionados) < limite:
                if not selecionados:
                    melhor = max(restantes, key=lambda i: sim_query[i])
                else:
                    def score(i):
                        div = max(_cos(cand_embs[i], cand_embs[j]) for j in selecionados)
                        return lambda_mult * sim_query[i] - (1 - lambda_mult) * div
                    melhor = max(restantes, key=score)
                selecionados.append(melhor)
                restantes.remove(melhor)
            return [candidatos[i] for i in selecionados]

        # Busca um lote amplo de candidatos (3x top-k, mínimo 24) para ter
        # material suficiente tanto para MMR quanto para a reserva de vagas CJF.
        fetch_k = max(top_k * 3, 24)
        res = collection.query(
            query_embeddings=[emb_query],
            n_results=fetch_k,
            include=["documents", "metadatas", "embeddings"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0]

        candidatos = [
            (doc, meta, emb)
            for doc, meta, emb in zip(docs, metas, embs)
            if doc is not None and str(doc).strip()
        ]
        if not candidatos:
            return []

        # ---- Priorização de jurisprudência (arquivos CJF) ------------------
        # Quando a pergunta pede decisões/julgados, reservamos vagas do top-k
        # para chunks CJF, evitando que os PDFs de legislação ocupem tudo.
        if self._pede_jurisprudencia(pergunta):
            n_pedido = self._num_julgados_solicitados(pergunta)
            # Reserva = nº pedido (com folga de +1) OU metade do top-k, limitado
            # ao próprio top-k.
            if n_pedido > 0:
                reserva_cjf = min(top_k, n_pedido + 1)
            else:
                reserva_cjf = max(1, top_k // 2)

            # Candidatos CJF presentes no lote geral.
            cand_cjf = [c for c in candidatos if self._e_chunk_cjf(c[1])]

            # Se o lote geral trouxe POUCOS CJF (os PDFs de legislação dominam a
            # similaridade), faz uma busca AMPLA e filtra os CJF dela. Garante
            # material de jurisprudência mesmo quando os CJF estão longe no
            # ranking global.
            if len(cand_cjf) < reserva_cjf:
                fetch_amplo = max(fetch_k, 300)
                res2 = collection.query(
                    query_embeddings=[emb_query],
                    n_results=fetch_amplo,
                    include=["documents", "metadatas", "embeddings"],
                )
                d2 = (res2.get("documents") or [[]])[0]
                m2 = (res2.get("metadatas") or [[]])[0]
                e2 = (res2.get("embeddings") or [[]])[0]
                # Dedup por conteúdo (evita repetir CJF já presente no lote geral).
                textos_cjf = {c[0] for c in cand_cjf}
                for doc, meta, emb in zip(d2, m2, e2):
                    if doc is None or not str(doc).strip():
                        continue
                    if self._e_chunk_cjf(meta) and doc not in textos_cjf:
                        cand_cjf.append((doc, meta, emb))
                        textos_cjf.add(doc)

            cand_outros = [c for c in candidatos if not self._e_chunk_cjf(c[1])]

            # Para a jurisprudência, prioriza ADERÊNCIA ao assunto da pergunta
            # (similaridade pura), não diversidade — assim os acórdãos trazidos
            # são os mais próximos do tema pedido.
            escolhidos_cjf = _mmr(cand_cjf, reserva_cjf, forcar_relevancia=True)
            # Completa o restante do top-k com os demais (legislação etc.),
            # onde a diversidade do MMR ainda é útil.
            vagas_restantes = top_k - len(escolhidos_cjf)
            escolhidos_outros = _mmr(cand_outros, vagas_restantes)

            selecao = (escolhidos_cjf + escolhidos_outros)[:top_k]
            return [
                {"conteudo": str(d), "metadata": m or {}}
                for d, m, _ in selecao
            ]

        # ---- Caso geral (sem pedido explícito de jurisprudência) -----------
        selecao = _mmr(candidatos, top_k)
        return [
            {"conteudo": str(d), "metadata": m or {}}
            for d, m, _ in selecao
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

    @staticmethod
    def _cabecalho_origem(meta: dict | None) -> str:
        """
        Reconstrói o cabeçalho de origem (arquivo/processo/tribunal) a partir
        dos METADADOS do chunk, para prefixar no contexto do RAG. Assim a LLM
        pode citar a decisão e o arquivo de onde ela veio, SEM que esse texto
        administrativo tenha entrado no embedding (o que prejudicava a busca).

        Compatível com índices antigos: se não houver os metadados de origem,
        cai no `source` (caminho do arquivo) ou retorna vazio.
        """
        import os as _os
        m = meta or {}
        arquivo = (
            m.get("arquivo_origem")
            or (_os.path.basename(str(m.get("source"))) if m.get("source") else "")
        )
        processo = m.get("processo_acordao") or m.get("PROCESSO JUDICIAL") or m.get("Nº")
        tribunal = m.get("tribunal_origem") or m.get("TRIBUNAL")

        partes = []
        if arquivo:
            partes.append(f"ARQUIVO DE ORIGEM: {arquivo}")
        if processo:
            partes.append(f"PROCESSO/ACÓRDÃO: {processo}")
        if tribunal:
            partes.append(f"TRIBUNAL: {tribunal}")
        return " | ".join(partes)

    def _montar_trecho_contexto(self, trecho: dict) -> str:
        """
        Monta o texto de um trecho para o CONTEXTO do RAG: prefixa o cabeçalho
        de origem (reconstruído dos metadados) + o conteúdo (ementa). O cabeçalho
        só é adicionado aqui (na geração), não no embedding.
        """
        cab = self._cabecalho_origem(trecho.get("metadata"))
        conteudo = str(trecho.get("conteudo", ""))
        return f"{cab}\n{conteudo}" if cab else conteudo

    def executar_rag(self, pergunta: str, on_wait=None) -> dict[str, Any]:
        """LLM com RAG: retrieval + geração, cronometrados separadamente."""
        # 1) Retrieval (embedding da pergunta + busca vetorial).
        # Busca direto na collection do Chroma (dicts crus), evitando que o
        # LangChain instancie Documents a partir de registros com texto nulo.
        t0 = time.perf_counter()
        trechos = self._recuperar_trechos(pergunta)
        tempo_retrieval = time.perf_counter() - t0

        # Monta o contexto PREFIXANDO o cabeçalho de origem (dos metadados) a
        # cada trecho, para a LLM poder citar processo/arquivo mesmo sem que esse
        # cabeçalho tenha entrado no embedding.
        contexto = "\n\n---\n\n".join(
            self._montar_trecho_contexto(t) for t in trechos
        )
        fontes = [
            {"conteudo": self._montar_trecho_contexto(t)[:500],
             "metadata": t["metadata"]}
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

def _montar_gabarito_estruturado(gab_item: dict) -> str:
    """
    Monta o texto do gabarito enviado ao Juiz, agregando — quando presentes —
    os campos ricos do dataset: resposta esperada, sentimento esperado,
    referências legais/jurisprudenciais, pontos obrigatórios e critérios
    inaceitáveis. Isso permite ao Juiz avaliar as respostas contra um gabarito
    completo, não apenas o texto-resumo.

    Campos ausentes são simplesmente omitidos (robusto a gabaritos simples).
    """
    if not gab_item:
        return ""

    partes: list[str] = []

    resposta = (
        gab_item.get("resposta_esperada")
        or gab_item.get("gabarito_oficial")
        or gab_item.get("resposta_referencia")
        or ""
    )
    if resposta:
        partes.append(f"RESPOSTA ESPERADA:\n{resposta}")

    sentimento = gab_item.get("sentimento_esperado")
    if sentimento:
        partes.append(f"SENTIMENTO ESPERADO: {sentimento}")

    julgados = gab_item.get("julgados_solicitados")
    if isinstance(julgados, int) and julgados > 0:
        partes.append(f"JULGADOS SOLICITADOS NA PERGUNTA: {julgados}")

    def _lista(campo: str, titulo: str):
        valores = gab_item.get(campo)
        if isinstance(valores, list) and valores:
            linhas = "\n".join(f"- {v}" for v in valores)
            partes.append(f"{titulo}:\n{linhas}")
        elif isinstance(valores, str) and valores.strip():
            partes.append(f"{titulo}:\n- {valores.strip()}")

    _lista("referencias", "REFERÊNCIA (legislação, súmulas, julgados e fontes)")
    _lista("pontos_obrigatorios", "PONTOS OBRIGATÓRIOS (devem constar na resposta)")
    _lista("criterios_inaceitaveis", "INACEITÁVEL (erros expressamente vedados)")

    ementa = gab_item.get("ementa_esperada")
    if isinstance(ementa, str) and ementa.strip():
        partes.append(f"EMENTA DE REFERÊNCIA:\n{ementa.strip()}")

    return "\n\n".join(partes)


# Termos que indicam, num erro crítico do Juiz, uma acusação de que a resposta
# INVENTOU um processo/julgado. Usados para suprimir esses erros quando os
# processos citados foram confirmados (DataJud ou base CJF).
_TERMOS_ALUCINACAO_PROC = (
    "inventou", "inventad", "alucin", "inexistente", "não existe", "nao existe",
    "fabricad", "fictício", "ficticio", "forjad", "número falso", "numero falso",
    "processo falso", "julgado falso", "não encontrado", "nao encontrado",
)
# Termos que indicam que o erro é SOBRE processos/julgados (e não outra coisa).
_TERMOS_REFERE_PROCESSO = (
    "processo", "julgado", "acórdão", "acordao", "decisão", "decisao",
    "jurisprud", "precedente", "ementa", "número cnj", "numero cnj",
)


def _ajustar_avaliacao_rag_por_validacao(
    aval_rag: dict,
    validacao_rag: list[dict],
    log: Callable[[str], None],
) -> None:
    """
    Ajusta DETERMINISTICAMENTE a avaliação do RAG usando a validação factual dos
    processos citados (DataJud + base CJF). Modifica `aval_rag` no lugar.

    Regras (aplicadas apenas ao RAG, que tem acesso à base):
      1. julgados_verificaveis: elevado para, no mínimo, o nº de processos
         citados CONFIRMADOS (existe no DataJud OU consta na base CJF). Nunca
         reduz o valor que o Juiz já atribuiu.
      2. Se há processos citados e TODOS os citados estão confirmados, remove de
         `erros_criticos` as entradas que acusam invenção/alucinação de processo
         (texto que combina um termo de "inventado/inexistente" com um termo que
         se refere a processo/julgado). Erros não relacionados a isso permanecem.
      3. Anexa à `justificativa_juiz` uma nota factual informando quantos
         processos citados foram confirmados (e quantos pela base CJF).
    """
    if not isinstance(aval_rag, dict) or not validacao_rag:
        return

    total = len(validacao_rag)
    confirmados = [p for p in validacao_rag if p.get("existe") or p.get("na_base_cjf")]
    n_conf = len(confirmados)
    n_base = sum(1 for p in confirmados if p.get("na_base_cjf") and not p.get("existe"))
    if n_conf == 0:
        return

    # 1) Eleva julgados_verificaveis (sem reduzir o valor do Juiz).
    verif_juiz = aval_rag.get("julgados_verificaveis") or 0
    try:
        verif_juiz = int(verif_juiz)
    except (TypeError, ValueError):
        verif_juiz = 0
    if n_conf > verif_juiz:
        aval_rag["julgados_verificaveis"] = n_conf
        log(f"   [ajuste:rag] julgados_verificaveis {verif_juiz} -> {n_conf} "
            f"(processos confirmados: {n_conf}/{total}, sendo {n_base} via base CJF).")

    # 2) Remove erros críticos de "alucinação de processo" quando TODOS os
    #    processos citados estão confirmados.
    todos_confirmados = (n_conf == total)
    if todos_confirmados:
        erros = aval_rag.get("erros_criticos") or []
        if isinstance(erros, list) and erros:
            def _e_falsa_alucinacao(texto: str) -> bool:
                t = str(texto).lower()
                refere_proc = any(k in t for k in _TERMOS_REFERE_PROCESSO)
                acusa_invencao = any(k in t for k in _TERMOS_ALUCINACAO_PROC)
                return refere_proc and acusa_invencao

            mantidos = [e for e in erros if not _e_falsa_alucinacao(e)]
            removidos = len(erros) - len(mantidos)
            if removidos:
                aval_rag["erros_criticos"] = mantidos
                log(f"   [ajuste:rag] {removidos} erro(s) crítico(s) de "
                    f"'alucinação de processo' removido(s): todos os {total} "
                    f"processo(s) citado(s) foram confirmados (DataJud/base CJF).")

    # 3) Nota factual na justificativa (sempre que houver confirmados).
    nota = (
        f"[Validação factual] {n_conf} de {total} processo(s) citado(s) "
        f"confirmado(s)"
    )
    if n_base:
        nota += f" ({n_base} localizado(s) na base CJF, não no DataJud)"
    nota += "."
    just = str(aval_rag.get("justificativa_juiz", "") or "").strip()
    aval_rag["justificativa_juiz"] = (just + " " + nota).strip() if just else nota


def _calcular_sentimento_aderente(
    sentimento_identificado: Any,
    sentimento_esperado: Any,
) -> bool | None:
    """
    Determina `sentimento_aderente_gabarito` comparando o sentimento
    IDENTIFICADO na resposta com o sentimento ESPERADO no gabarito, seguindo
    regras rígidas (determinístico, não depende do booleano do modelo):

      1. Correspondência: identificado == esperado  -> True.
      2. Divergência: identificado exclusivamente oposto ao esperado -> False.
      3. Múltiplos sentimentos: se a resposta apresenta positivo E negativo,
         -> True quando o gabarito indicar qualquer um desses.
      4. Gabarito sem sentimento claro (positivo/negativo) -> None.
      5. (equivalente a 4 no plano de dados) sem requisito/sem esperado -> None.
      6. Coerência: nunca False se houver correspondência de ao menos um dos
         sentimentos identificados com o esperado.

    Categorias consideradas: positivo, negativo, neutro. Normaliza caixa/espaços.
    """
    def _norm(v: Any) -> str:
        return str(v or "").strip().lower()

    esperado = _norm(sentimento_esperado)
    identificado = _norm(sentimento_identificado)

    # Regras 4 e 5: só avaliamos aderência quando o GABARITO define um
    # sentimento claro (positivo ou negativo). Caso contrário, None.
    if esperado not in ("positivo", "negativo"):
        return None

    tem_pos = "positivo" in identificado
    tem_neg = "negativo" in identificado

    # Regra 3 + 6: múltiplos sentimentos (positivo E negativo) identificados.
    # Como o esperado é positivo ou negativo, qualquer um deles está presente
    # -> há correspondência -> True.
    if tem_pos and tem_neg:
        return True

    # Regra 1: correspondência exata com o esperado.
    if esperado == "positivo" and tem_pos:
        return True
    if esperado == "negativo" and tem_neg:
        return True

    # Regra 2: oposto exclusivo do esperado.
    if esperado == "positivo" and tem_neg:
        return False
    if esperado == "negativo" and tem_pos:
        return False

    # Identificado é neutro/inconclusivo/ausente: não corresponde ao esperado
    # claro do gabarito -> divergente (o gabarito exigia um sentimento definido).
    return False


def _e_limite_diario_tokens(exc: Exception) -> bool:
    """
    Detecta o erro de LIMITE DIÁRIO de tokens da Groq (TPD - tokens per day),
    que NÃO é resolvido por backoff: a cota do dia acabou e novas chamadas
    continuarão falhando até o reset. Identificado pela mensagem/código do erro.
    """
    txt = str(exc).lower()
    return (
        "rate_limit_exceeded" in txt
        or "tokens per day" in txt
        or "tpd" in txt
        or ("limit" in txt and "per day" in txt)
        or ("429" in txt and "day" in txt)
    )


def executar_benchmarking(
    config: Config,
    perguntas: list[dict],
    gabarito: list[dict],
    modelo_execucao: str | None = None,
    modelo_juiz: str | None = None,
    log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, dict], None] | None = None,
    reindexar: bool = False,
    arquivos_selecionados: list[str] | None = None,
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
        reindexar: se True, ZERA a base e reindexa antes de rodar.
        arquivos_selecionados: nomes dos arquivos a reindexar quando
            reindexar=True. Se vazio/None, NÃO reindexa nada (usa a base atual).

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
    base_dir = config.dados["base_juridica"]
    if reindexar:
        # "Reindexar do zero ao executar": zera a base e reindexa APENAS os
        # arquivos selecionados. Se nenhum foi selecionado, NÃO indexa nada
        # (mantém o que já estiver no banco).
        selecionados = arquivos_selecionados or []
        if selecionados:
            from pathlib import Path as _P
            base_p = _P(base_dir)
            caminhos = []
            for nome in selecionados:
                p = _P(nome)
                if not p.exists():
                    achados = list(base_p.glob(f"**/{nome}"))
                    p = achados[0] if achados else p
                caminhos.append(str(p))
            log(f"[rag] Reindexar do zero: zerando base e reindexando "
                f"{len(caminhos)} arquivo(s) selecionado(s).")
            pipeline.indexar_selecionados(caminhos, log=log, zerar=True)
        else:
            log("[rag] 'Reindexar do zero' marcado, mas nenhum arquivo "
                "selecionado — nada será reindexado; usando a base atual.")
            pipeline.indexar_base(base_dir, log=log, forcar=False)
    else:
        # Sem reindexar: apenas carrega a base existente.
        pipeline.indexar_base(base_dir, log=log, forcar=False)

    gab_idx = indexar_por_id(gabarito)
    total = len(perguntas)
    resultados: list[dict] = []
    # Preenchido quando a execução é interrompida no meio (ex.: limite diário de
    # tokens da Groq). Guarda a mensagem para informar o usuário e salvar os
    # parciais já processados.
    interrompido: dict[str, Any] | None = None

    for i, p in enumerate(perguntas, start=1):
        pid = str(p.get("id", p.get("id_pergunta", i)))
        pergunta = p.get("pergunta", "")
        gab_item = gab_idx.get(pid, {})
        gab_texto = _montar_gabarito_estruturado(gab_item)

        log(f"\n> Pergunta {i}/{total} ({pid}): {pergunta[:60]}...")

        def _wait_log(espera, tentativa):
            log(f"   [429] Rate limit. Backoff {espera:.1f}s (tentativa {tentativa})...")

        try:
            # LLM Padrão
            r_padrao = pipeline.executar_padrao(pergunta, on_wait=_wait_log)
            log(f"   [padrao] LLM Padrão respondeu em "
                f"{r_padrao['tempo_execucao_padrao_s']}s")

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
        except Exception as exc:  # noqa: BLE001
            # Falha nas chamadas às LLMs. Se for o LIMITE DIÁRIO de tokens (TPD),
            # não adianta continuar — a cota do dia acabou. Interrompemos o loop
            # PRESERVANDO os resultados já processados (serão salvos e exibidos).
            if _e_limite_diario_tokens(exc):
                log(f"\n[LIMITE] Limite diário de tokens da Groq atingido na "
                    f"pergunta {i}/{total} ({pid}). Interrompendo e salvando o "
                    f"que já foi processado ({len(resultados)} pergunta(s)).")
                interrompido = {
                    "motivo": "limite_diario_tokens",
                    "mensagem": (
                        "Limite diário de tokens da Groq (TPD) atingido. A "
                        "execução foi interrompida e os resultados processados "
                        "até aqui foram salvos. Tente novamente após o reset da "
                        "cota (ou use outra chave/plano)."
                    ),
                    "pergunta_parada": pid,
                    "indice_parada": i,
                    "total_perguntas": total,
                    "processadas": len(resultados),
                    "detalhe_erro": str(exc),
                }
                break
            # Outros erros numa pergunta: registra e PULA para a próxima, sem
            # abortar o benchmark inteiro (preserva robustez).
            log(f"   [erro] Falha na pergunta {pid}: {exc}. Pulando para a próxima.")
            continue

        # Reconcilia a aderência de sentimento de forma DETERMINÍSTICA (ver
        # _calcular_sentimento_aderente): compara o sentimento IDENTIFICADO na
        # resposta com o sentimento ESPERADO do gabarito, aplicando as regras de
        # correspondência/divergência/múltiplos sentimentos. Não confia no
        # booleano que o modelo devolve (que às vezes se contradiz). Quando o
        # gabarito não define um sentimento claro, o resultado é None.
        sent_esperado = gab_item.get("sentimento_esperado")
        for chave in ("avaliacao_llm_padrao", "avaliacao_llm_rag"):
            av = aval.get(chave) or {}
            # Falha técnica de avaliação não tem sentimento confiável: mantém o
            # que veio (geralmente None) e não força aderência.
            if str(av.get("status_avaliacao", "valida")) != "valida":
                continue
            av["sentimento_aderente_gabarito"] = _calcular_sentimento_aderente(
                av.get("sentimento_identificado"), sent_esperado
            )

        def _fmt_notas(rotulo: str, av: dict):
            erros = av.get("erros_criticos") or []
            n_erros = len(erros) if isinstance(erros, list) else 0
            sent = av.get("sentimento_identificado", "?")
            aderente = av.get("sentimento_aderente_gabarito")
            aderente_txt = (
                "aderente" if aderente is True
                else "divergente" if aderente is False
                else "n/a"
            )
            log(
                f"   [juiz:{rotulo}] fidelidade={av.get('nota_fidelidade')}/5 | "
                f"precisão={av.get('nota_precisao_gabarito')}/5 | "
                f"completude={av.get('nota_completude')}/5 | "
                f"sentimento={sent} ({aderente_txt}) | "
                f"julgados={av.get('julgados_apresentados')}/"
                f"{av.get('julgados_solicitados')} "
                f"(verif. {av.get('julgados_verificaveis')}) | "
                f"erros_críticos={n_erros}"
            )
            for e in (erros if isinstance(erros, list) else []):
                log(f"   [juiz:{rotulo}] erro crítico: {e}")
            justificativa = av.get("justificativa_juiz")
            if justificativa:
                log(f"   [juiz:{rotulo}] justificativa: {justificativa}")

        _fmt_notas("padrao", aval["avaliacao_llm_padrao"])
        _fmt_notas("rag", aval["avaliacao_llm_rag"])

        # Validação factual de processos citados (DataJud - TRFs), se habilitada
        if validar_datajud:
            palavras_tema = config.tema.get("palavras_chave_datajud")
            # A LLM Padrão NÃO tem acesso à base; um processo não encontrado no
            # DataJud é de fato suspeito de invenção. Por isso não passamos
            # base_dir para o Padrão.
            validacao_padrao = validar_processos_citados(
                r_padrao["resposta"], palavras_tema=palavras_tema
            )
            # Para o RAG, passamos base_dir: se o DataJud não achar mas o número
            # constar nos CJF, confirma-se que a citação veio da base (não é
            # alucinação) via os campos na_base_cjf/arquivo_cjf.
            validacao_rag = validar_processos_citados(
                r_rag["resposta"],
                palavras_tema=palavras_tema,
                base_dir=base_dir,
            )
        else:
            validacao_padrao = []
            validacao_rag = []

        def _log_validacao(rotulo: str, procs: list[dict]):
            if not procs:
                return
            log(f"   [datajud:{rotulo}] {len(procs)} processo(s) citado(s):")
            for p in procs:
                na_base = p.get("na_base_cjf")
                arq_base = p.get("arquivo_cjf")
                if p.get("existe"):
                    compat = "compatível" if p.get("compativel_tema") else "INCOMPATÍVEL"
                    assuntos = ", ".join(p.get("assuntos", [])) or "(sem assunto)"
                    log(
                        f"      - {p['numero']} -> existe | assunto: {assuntos} "
                        f"| tema: {compat}"
                    )
                elif na_base:
                    # Número consta na base CJF: citação legítima. Ou o DataJud
                    # foi dispensado (preferir base), ou não confirmou (erro/sem
                    # hits) mas a base confirma.
                    if p.get("datajud_dispensado"):
                        log(f"      - {p['numero']} -> consta na base CJF "
                            f"({arq_base}); DataJud dispensado")
                    else:
                        motivo = p.get("erro") or "não localizado no DataJud"
                        log(f"      - {p['numero']} -> {motivo}, MAS consta na "
                            f"base CJF ({arq_base})")
                elif p.get("erro"):
                    log(f"      - {p['numero']} -> {p['erro']}")
                else:
                    log(f"      - {p['numero']} -> NÃO ENCONTRADO no DataJud nem "
                        f"na base CJF (possível alucinação)")

        _log_validacao("padrao", validacao_padrao)
        _log_validacao("rag", validacao_rag)

        # Ajuste determinístico da avaliação do RAG com base na validação
        # factual: processos citados confirmados (DataJud ou base CJF) contam
        # como verificáveis e não devem ser tratados como invenção pelo Juiz.
        if validar_datajud:
            _ajustar_avaliacao_rag_por_validacao(
                aval["avaliacao_llm_rag"], validacao_rag, log
            )

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
    # Se a execução foi INTERROMPIDA (ex.: limite diário de tokens), NÃO tentamos
    # gerar o parecer — gastaria mais tokens e também falharia.
    parecer = {}
    if resultados and interrompido is None:
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
    elif interrompido is not None:
        log("[juiz] Parecer final NÃO gerado (execução interrompida).")

    return montar_relatorio(
        resultados,
        modelo_testado=modelo_execucao or config.modelos["llm_execucao"],
        modelo_juiz=modelo_juiz or config.modelos["llm_juiz"],
        tempos_indexacao=pipeline.tempos_indexacao,
        parecer_final=parecer,
        interrupcao=interrompido,
    )
