"""
Interface de usuário (Streamlit) do llm-tester.

Organizada em abas:
  1. Configurações (API, Base Jurídica, Modelos, Parâmetros)
  2. Dados de Teste (Perguntas e Gabaritos)
  3. Benchmarking (Execução Individual e Execução em Lote)
  4. Dashboard de Resultados (analytics & export)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import Config
from .pipeline import executar_benchmarking
from . import utils


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

def _init_state():
    ss = st.session_state
    ss.setdefault("config", Config("config.json"))
    ss.setdefault("perguntas", None)
    ss.setdefault("gabarito", None)
    ss.setdefault("base_dir", ss["config"].dados["base_juridica"])
    ss.setdefault("relatorio", None)
    ss.setdefault("logs", [])
    ss.setdefault("resultados_parciais", [])
    ss.setdefault("executando", False)
    ss.setdefault("iniciar_lote", False)
    ss.setdefault("lote_selecionadas", [])

    # Limpa o log de reindexação de execuções ANTERIORES uma única vez por
    # sessão, para a aba de Reindexação começar limpa (não mostrar o resultado
    # de uma reindexação de outra sessão). Só roda na primeira carga.
    if not ss.get("_log_reindex_limpo"):
        try:
            Path("_reindex_log.txt").unlink(missing_ok=True)
        except OSError:
            pass
        ss["_log_reindex_limpo"] = True

    # Aplica o tema configurado ao metadata dos datasets salvos.
    tema = ss["config"].tema
    utils.configurar_tema(tema.get("nome"), tema.get("area"))


def _add_log(msg: str):
    st.session_state["logs"].append(msg)


def _render_logs(container, linhas: list[str], altura: int = 320) -> None:
    """
    Renderiza os logs numa área rolável com quebra de linha, evitando que o
    conteúdo seja cortado à direita (como acontece com st.code).

    - `container`: um st.empty() (streaming) ou o próprio st.
    - `altura`: altura da caixa em pixels (scroll vertical quando excede).
    """
    from html import escape

    texto = "\n".join(linhas) if linhas else "(sem logs)"
    html = (
        f'<div style="height:{altura}px; overflow:auto; '
        'background:#0e1117; color:#d0d0d0; border:1px solid #333; '
        'border-radius:6px; padding:10px; font-family:monospace; '
        'font-size:12px; white-space:pre-wrap; word-break:break-word;">'
        f'{escape(texto)}</div>'
    )
    container.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Aba 1 — Configuração
# ---------------------------------------------------------------------------

def _validar_diretorio_base(caminho: str) -> None:
    """
    Feedback amigável para o diretório da base jurídica.

    Aceita caminhos locais e pastas de nuvem sincronizadas (montadas como
    diretório no sistema de arquivos). Alerta em casos comuns de erro:
    campo vazio, URL/link de nuvem, caminho inexistente ou sem PDFs/TXTs/CSVs.
    """
    texto = (caminho or "").strip()
    if not texto:
        st.warning("Informe um diretório para a base jurídica.")
        return

    # Não suportamos acesso via API/URL — apenas caminhos do sistema de arquivos.
    if texto.lower().startswith(("http://", "https://", "ftp://")):
        st.error(
            "URLs não são suportadas. Use um caminho de pasta local ou de nuvem "
            "sincronizada (ex.: uma pasta do OneDrive/Google Drive no seu disco)."
        )
        return

    caminho_p = Path(texto)
    if not caminho_p.exists():
        st.warning(
            f"O diretório '{texto}' ainda não existe. Ele será criado ao enviar "
            "arquivos, ou informe um caminho já existente."
        )
        return
    if not caminho_p.is_dir():
        st.error(f"O caminho '{texto}' existe, mas não é um diretório.")
        return

    n_arquivos = len(list(caminho_p.glob("**/*.pdf"))) + len(
        list(caminho_p.glob("**/*.txt"))
    ) + len(list(caminho_p.glob("**/*.csv")))
    if n_arquivos == 0:
        st.warning(
            "Diretório encontrado, mas sem arquivos PDF/TXT/CSV. Adicione documentos "
            "ou envie-os abaixo."
        )
    else:
        st.caption(f"✅ {n_arquivos} arquivo(s) PDF/TXT/CSV encontrado(s) em '{texto}'.")


def _painel_tema(cfg: Config) -> None:
    """Painel de configuração do tema (domínio) na aba Configurações."""
    st.subheader("🏷️ Tema")
    st.caption(
        "Define o domínio da aplicação. Aplicado à sessão imediatamente; use "
        "o botão abaixo para gravar no config.json."
    )

    tema = cfg.data.setdefault("tema", {})
    nome_original = tema.get("nome", "")

    col1, col2 = st.columns(2)
    with col1:
        nome = st.text_input("Nome do tema", value=tema.get("nome", ""))
        dominio = st.text_input(
            "Domínio (usado nos prompts das LLMs)",
            value=tema.get("dominio_prompt", ""),
            help="Preenche o placeholder {dominio} dos prompts Padrão e RAG.",
        )
    with col2:
        area = st.text_input("Área", value=tema.get("area", ""))
        especialidade = st.text_input(
            "Especialidade do Juiz",
            value=tema.get("especialidade_juiz", ""),
            help="Especialidade declarada no system prompt da LLM Juiz.",
        )

    palavras_txt = st.text_input(
        "Palavras-chave DataJud (separadas por vírgula)",
        value=", ".join(tema.get("palavras_chave_datajud", []) or []),
        help=(
            "Usadas para checar se o assunto do processo consultado no DataJud "
            "é compatível com o tema."
        ),
    )
    palavras = [p.strip() for p in palavras_txt.split(",") if p.strip()]

    # Aplica na sessão imediatamente.
    tema["nome"] = nome
    tema["area"] = area
    tema["dominio_prompt"] = dominio
    tema["especialidade_juiz"] = especialidade
    tema["palavras_chave_datajud"] = palavras
    utils.configurar_tema(nome, area)

    # Aviso: trocar o tema pede reindexação para não misturar embeddings.
    if nome and nome != nome_original:
        st.warning(
            "Você alterou o nome do tema. Marque **Reindexar base do zero** "
            "(ou troque o `collection_name` em Parâmetros) para não misturar os "
            "embeddings do tema anterior com os do novo no ChromaDB."
        )

    if st.button("💾 Salvar tema no config.json"):
        try:
            destino = cfg.salvar()
            st.success(f"Tema salvo em {destino}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao salvar o config.json: {exc}")


def _aba_configuracao():
    st.header("⚙️ Configurações")
    cfg: Config = st.session_state["config"]

    st.subheader("🔑 API")
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input(
            "GROQ_API_KEY",
            value=cfg.groq_api_key or "",
            type="password",
            help="Chave da Groq API (free tier).",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔌 Validar conexão"):
            _validar_groq(api_key)
    if api_key:
        cfg.set_groq_api_key(api_key)

    st.divider()
    _painel_tema(cfg)

    st.divider()
    st.subheader("📚 Base Jurídica")
    base_dir = st.text_input(
        "Diretório da base jurídica (PDFs/TXTs/CSVs)",
        value=st.session_state["base_dir"],
        help=(
            "Aceita um diretório local ou uma pasta de nuvem sincronizada "
            "(OneDrive, Google Drive, Dropbox) ou unidade de rede mapeada "
            "(ex.: Z:\\base ou \\\\servidor\\compartilhamento). Não aceita "
            "URLs (http/https) nem links de nuvem."
        ),
    )
    st.session_state["base_dir"] = base_dir
    _validar_diretorio_base(base_dir)
    up_docs = st.file_uploader(
        "Ou envie documentos (PDFs/TXTs/CSVs)", type=["pdf", "txt", "csv"], accept_multiple_files=True
    )
    arquivos_enviados = []
    if up_docs:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        for doc in up_docs:
            (Path(base_dir) / doc.name).write_bytes(doc.getvalue())
            arquivos_enviados.append(doc.name)
        st.success(f"{len(up_docs)} arquivo(s) salvos em {base_dir}")
        # Atalho: indexar imediatamente só os arquivos recém-enviados.
        if st.button("⚡ Indexar arquivos enviados agora", type="primary"):
            for nome in arquivos_enviados:
                _disparar_reindexacao(nome)
            st.success(
                f"Indexação de {len(arquivos_enviados)} arquivo(s) iniciada em "
                "segundo plano. Acompanhe o progresso na seção de reindexação abaixo."
            )

    # Delimitador dos CSVs da base jurídica.
    csv_cfg = cfg.parametros.setdefault("csv_base_juridica", {})
    opcoes_delim = {
        "Ponto e vírgula ( ; )": ";",
        "Vírgula ( , )": ",",
        "Tabulação ( tab )": "\t",
        "Detectar automaticamente": "auto",
    }
    delim_atual = csv_cfg.get("delimitador", ";")
    labels = list(opcoes_delim.keys())
    valores = list(opcoes_delim.values())
    idx_atual = valores.index(delim_atual) if delim_atual in valores else 0
    delim_label = st.selectbox(
        "Delimitador dos arquivos CSV",
        labels,
        index=idx_atual,
        help=(
            "Separador de colunas usado nos CSVs da base jurídica. Use ';' "
            "quando o texto contém muitas vírgulas (ex.: ementas). "
            "'Detectar automaticamente' tenta identificar sozinho."
        ),
    )
    csv_cfg["delimitador"] = opcoes_delim[delim_label]

    reindexar = st.checkbox(
        "Reindexar base do zero ao executar",
        value=False,
        help="Quando marcado, ao executar o benchmark a base é ZERADA e "
             "reindexada apenas com os arquivos selecionados abaixo. Se nenhum "
             "arquivo estiver selecionado, nada é reindexado (usa a base atual).",
    )
    st.session_state["reindexar"] = reindexar

    # Seleção de arquivos + indexação imediata. A mesma seleção é usada pelo
    # benchmark quando 'Reindexar base do zero ao executar' está marcado.
    with st.expander("🔁 Seleção de arquivos e indexação", expanded=False):
        _secao_reindexacao(cfg, contexto="config")

    st.divider()
    st.subheader("🧠 Modelos (Groq API)")
    modelos = cfg.modelos["modelos_disponiveis"]
    col_a, col_b = st.columns(2)
    with col_a:
        modelo_exec = st.selectbox(
            "LLM de Execução",
            modelos,
            index=modelos.index(cfg.modelos["llm_execucao"])
            if cfg.modelos["llm_execucao"] in modelos else 0,
        )
    with col_b:
        modelo_juiz = st.selectbox(
            "LLM Juiz (recomendado: llama-3.3-70b-versatile)",
            modelos,
            index=modelos.index(cfg.modelos["llm_juiz"])
            if cfg.modelos["llm_juiz"] in modelos else 0,
        )
    st.session_state["modelo_exec"] = modelo_exec
    st.session_state["modelo_juiz"] = modelo_juiz

    st.divider()
    st.subheader("⚙️ Parâmetros")
    col_t, col_c = st.columns(2)
    with col_t:
        temp = st.slider("Temperature", 0.0, 1.0, float(cfg.parametros["temperature"]), 0.1)
    with col_c:
        chunk = st.slider("Chunk size (tokens)", 200, 2000, int(cfg.parametros["chunk_size"]), 50)
    cfg.parametros["temperature"] = temp
    cfg.parametros["chunk_size"] = chunk

    validar_datajud = st.checkbox(
        "Validar processos citados no DataJud (TRFs)",
        value=bool(cfg.parametros.get("validar_processos_datajud", True)),
        help=(
            "Consulta a API Pública do DataJud (CNJ) para verificar se os "
            "números de processo citados existem e se o assunto é compatível "
            "com o tema. Pode adicionar latência, pois a API é lenta."
        ),
    )
    cfg.parametros["validar_processos_datajud"] = validar_datajud


# ---------------------------------------------------------------------------
# Aba 2 — Dados de Teste (Perguntas e Gabaritos)
# ---------------------------------------------------------------------------

def _editor_dataset(
    titulo: str,
    dados_key: str,
    colunas: list[str],
    carregar_bytes,
    carregar_arquivo,
    salvar_fn,
    caminho: str,
):
    """
    Editor genérico de dataset (perguntas/gabarito) com:
      - upload de arquivo
      - edição inline (adicionar / editar / excluir linhas)
      - exclusão explícita via checkbox + botão
      - salvar no arquivo padrão
    """
    ss = st.session_state
    st.subheader(titulo)

    up = st.file_uploader(
        f"Carregar {titulo} (.json / .csv / .xlsx)",
        type=["json", "csv", "xlsx"], key=f"up_{dados_key}"
    )
    if up is not None:
        try:
            itens = carregar_bytes(up.getvalue(), up.name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao ler '{up.name}': {exc}")
            itens = []
        if itens:
            ss[dados_key] = itens
            st.success(f"{len(itens)} item(ns) carregado(s).")
        else:
            st.warning(
                f"Nenhum item reconhecido em '{up.name}'. Verifique se o JSON tem "
                f"a chave '{dados_key}' (ou é uma lista), ou se o CSV/XLSX tem "
                "as colunas esperadas."
            )

    # Carrega o dataset padrão se ainda não houver nada em sessão.
    if not ss.get(dados_key):
        try:
            ss[dados_key] = carregar_arquivo(caminho)
        except Exception:  # noqa: BLE001
            ss[dados_key] = []

    # Monta o DataFrame garantindo todas as colunas esperadas.
    df = pd.DataFrame(ss.get(dados_key) or [])
    for c in colunas:
        if c not in df.columns:
            df[c] = None
    # Mantém colunas conhecidas primeiro, preservando eventuais extras.
    extras = [c for c in df.columns if c not in colunas]
    df = df[colunas + extras]

    # Coluna de seleção para exclusão
    df.insert(0, "🗑️", False)

    st.caption(
        "Edite as células, adicione linhas (＋) ou marque a coluna 🗑️ e clique "
        "'Excluir selecionados'."
    )
    df_editado = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor_{dados_key}",
        column_config={
            "🗑️": st.column_config.CheckboxColumn("🗑️", default=False, width="small"),
        },
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("💾 Salvar", key=f"salvar_{dados_key}"):
            registros = _extrair_registros(df_editado, colunas)
            ss[dados_key] = registros
            salvar_fn(registros, caminho)
            st.success(f"Salvo em {caminho} ({len(registros)} item(ns)).")
    with col_b:
        if st.button("🗑️ Excluir selecionados", key=f"excluir_{dados_key}"):
            mask = df_editado["🗑️"].fillna(False).astype(bool)
            n_excluir = mask.sum()
            if n_excluir == 0:
                st.warning("Nenhum item marcado para exclusão.")
            else:
                df_filtrado = df_editado[~mask]
                registros = _extrair_registros(df_filtrado, colunas)
                ss[dados_key] = registros
                salvar_fn(registros, caminho)
                st.success(
                    f"{n_excluir} item(ns) excluído(s). "
                    f"Restam {len(registros)} item(ns)."
                )
                st.rerun()
    with col_c:
        st.caption(f"Total atual: {len(df_editado)} linha(s).")


def _extrair_registros(df: "pd.DataFrame", colunas: list[str]) -> list[dict]:
    """Extrai registros válidos de um DataFrame editado, removendo linhas vazias."""
    # Remove a coluna de seleção para exclusão se presente.
    cols_saida = [c for c in df.columns if c != "🗑️"]

    def _vazio(v) -> bool:
        """True se o valor deve ser descartado. Robusto a listas/arrays."""
        if isinstance(v, (list, tuple)):
            return len(v) == 0
        try:
            if pd.isna(v):
                return True
        except (TypeError, ValueError):
            # pd.isna sobre tipos não escalares pode falhar: trata como não-vazio.
            return False
        return v == ""

    registros = [
        {k: v for k, v in row.items() if k in cols_saida and not _vazio(v)}
        for row in df.to_dict(orient="records")
    ]
    return [r for r in registros if r]


def _aba_dados_teste():
    st.header("🗂️ Dados")
    st.caption(
        "Cadastre e faça manutenção do Gabarito (Perguntas/Respostas). As "
        "perguntas usadas na execução são extraídas automaticamente deste "
        "mesmo arquivo — não há carga separada de perguntas. Sem dados "
        "carregados, o sistema usa o arquivo padrão em data/."
    )
    cfg: Config = st.session_state["config"]

    _editor_dataset(
        titulo="✅ Gabarito (Perguntas/Respostas)",
        dados_key="gabarito",
        colunas=["id", "pergunta", "resposta_esperada", "ementa_esperada",
                 "referencias", "criterios_inaceitaveis", "sentimento_esperado",
                 "julgados_solicitados"],
        carregar_bytes=utils.carregar_gabarito_bytes,
        carregar_arquivo=utils.carregar_gabarito,
        salvar_fn=utils.salvar_gabarito,
        caminho=cfg.dados["arquivo_gabarito"],
    )


def _validar_groq(api_key: str):
    if not api_key:
        st.error("Informe a chave GROQ_API_KEY.")
        return
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        cfg: Config = st.session_state["config"]
        llm = ChatGroq(
            model=cfg.modelos["llm_execucao"], api_key=api_key, max_tokens=5, max_retries=0
        )
        llm.invoke([HumanMessage(content="ping")])
        st.success("✅ Conexão com a Groq API validada.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha na conexão: {exc}")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def _iniciar_benchmarking(
    perguntas=None,
    gabarito=None,
    log_box=None,
    tabela_box=None,
    progress_box=None,
):
    ss = st.session_state
    cfg: Config = ss["config"]

    if not cfg.groq_api_key:
        st.error("GROQ_API_KEY não configurada.")
        return

    if gabarito is None:
        gabarito = ss["gabarito"] or utils.carregar_gabarito(cfg.dados["arquivo_gabarito"])
    # As perguntas são DERIVADAS do gabarito (fonte única). Só caímos no derive
    # completo quando o chamador não passou um subconjunto selecionado.
    if perguntas is None:
        perguntas = utils.perguntas_do_gabarito(gabarito)

    if not perguntas:
        st.warning("Carregue o Gabarito na aba 'Dados' antes de executar.")
        return

    cfg.dados["base_juridica"] = ss["base_dir"]

    ss["logs"] = []
    ss["resultados_parciais"] = []
    ss["relatorio"] = None
    ss["executando"] = True

    # Usa os placeholders fornecidos (posicionados na ordem correta pela aba);
    # se não vierem, cria localmente como fallback.
    log_box = log_box or st.empty()
    tabela_box = tabela_box or st.empty()
    progress_box = progress_box or st.empty()
    progress = progress_box.progress(0, text="Iniciando...")

    def log(msg: str):
        _add_log(msg)
        _render_logs(log_box, ss["logs"][-40:])

    def on_progress(i, total, resultado):
        progress.progress(i / total, text=f"Processando pergunta {i} de {total}...")
        ss["resultados_parciais"].append(resultado)
        tabela_box.dataframe(_df_parcial(ss["resultados_parciais"]), use_container_width=True)

    try:
        relatorio = executar_benchmarking(
            config=cfg,
            perguntas=perguntas,
            gabarito=gabarito,
            modelo_execucao=ss.get("modelo_exec"),
            modelo_juiz=ss.get("modelo_juiz"),
            log=log,
            on_progress=on_progress,
            reindexar=ss.get("reindexar", False),
            arquivos_selecionados=ss.get("reindex_selecao_config", []),
        )
        ss["relatorio"] = relatorio
        # Persiste em outputs/
        utils.salvar_json(relatorio, cfg.dados["diretorio_saida"])
        utils.salvar_csv(relatorio, cfg.dados["diretorio_saida"])
        progress.progress(1.0, text="Concluído!")
        st.success("✅ Execução concluída. Veja o Dashboard de Resultados.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro durante a execução: {exc}")
    finally:
        ss["executando"] = False


def _df_parcial(resultados: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        linhas.append({
            "ID": r["id_pergunta"],
            # Texto completo: o st.dataframe trunca visualmente na célula, mas o
            # usuário pode alargar a coluna ou passar o mouse para ver tudo.
            "Pergunta": r["pergunta"],
            "Tempo LLM Padrão (s)": f"{r['tempo_execucao_padrao_s']:.4f}",
            "Tempo LLM RAG (s)": f"{r['tempo_total_rag_s']:.4f}",
            "Nota LLM Padrão": r["avaliacao_llm_padrao"].get("nota_precisao_gabarito"),
            "Nota LLM RAG": r["avaliacao_llm_rag"].get("nota_precisao_gabarito"),
        })
    return pd.DataFrame(linhas)


def _executar_individual(pergunta: str, gabarito: str):
    """Executa uma única pergunta e exibe o resultado localmente (sem Dashboard)."""
    ss = st.session_state
    cfg: Config = ss["config"]

    if not cfg.groq_api_key:
        st.error("GROQ_API_KEY não configurada.")
        return
    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
        return

    cfg.dados["base_juridica"] = ss["base_dir"]
    perguntas = [{"id": "AVULSA", "pergunta": pergunta}]
    gab = [{"id": "AVULSA", "gabarito_oficial": gabarito or ""}]

    logs_ind: list[str] = []
    log_box = st.empty()

    def log(msg: str):
        logs_ind.append(msg)
        _render_logs(log_box, logs_ind[-40:])

    with st.spinner("Executando (Padrão + RAG + Juiz)..."):
        try:
            relatorio = executar_benchmarking(
                config=cfg,
                perguntas=perguntas,
                gabarito=gab,
                modelo_execucao=ss.get("modelo_exec"),
                modelo_juiz=ss.get("modelo_juiz"),
                log=log,
                reindexar=ss.get("reindexar", False),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erro durante a execução: {exc}")
            return

    resultados = relatorio.get("resultados_detalhados", [])
    if resultados:
        st.success("✅ Execução concluída.")

        r = resultados[0]
        st.subheader("💬 Respostas geradas")
        col_p, col_r = st.columns(2)
        with col_p:
            st.markdown("**🔵 LLM Padrão**")
            st.caption(f"Tempo: {r.get('tempo_execucao_padrao_s', 0):.4f} s")
            st.write(r.get("resposta_llm_padrao", "") or "(sem resposta)")
        with col_r:
            st.markdown("**🟣 LLM com RAG**")
            st.caption(f"Tempo: {r.get('tempo_total_rag_s', 0):.4f} s")
            st.write(r.get("resposta_llm_rag", "") or "(sem resposta)")

        _secao_detalhamento_juiz(relatorio)


def _aba_benchmarking():
    st.header("▶️ Execução")
    ss = st.session_state
    cfg: Config = ss["config"]

    sub_lote, sub_ind = st.tabs(["📚 Execução em Lote", "📄 Execução Individual"])

    # ---- Execução em Lote -----------------------------------------------
    with sub_lote:
        # Perguntas derivadas do Gabarito (fonte única). Carrega o gabarito da
        # sessão ou do arquivo padrão, e extrai as perguntas dele.
        gabarito_sessao = ss.get("gabarito")
        if not gabarito_sessao:
            try:
                gabarito_sessao = utils.carregar_gabarito(cfg.dados["arquivo_gabarito"])
            except Exception:  # noqa: BLE001
                gabarito_sessao = []
        perguntas = utils.perguntas_do_gabarito(gabarito_sessao)
        if not perguntas:
            st.info("Carregue o Gabarito na aba 'Dados'.")
        else:
            st.caption(
                "Marque as perguntas que deseja executar. Use o controle abaixo "
                "para ver o enunciado completo ou apenas um resumo."
            )

            item_keys = [f"lote_p_{idx}" for idx in range(len(perguntas))]

            # Inicializa o estado dos checkboxes na primeira renderização.
            ss.setdefault("lote_todas", True)
            for k in item_keys:
                ss.setdefault(k, ss["lote_todas"])

            def _on_toggle_todas():
                for k in item_keys:
                    ss[k] = ss["lote_todas"]

            def _on_toggle_item():
                ss["lote_todas"] = all(ss[k] for k in item_keys)

            # Linha de controles compactos: selecionar todas + exibir completa.
            cc1, cc2, _cc = st.columns([2, 2, 6])
            with cc1:
                st.checkbox(
                    "Selecionar todas", key="lote_todas",
                    on_change=_on_toggle_todas,
                )
            with cc2:
                mostrar_completa = st.checkbox(
                    "Mostrar pergunta completa", key="lote_mostrar_completa"
                )

            # Limite de caracteres exibidos quando não está em modo "completa".
            LIMITE = 140

            selecionadas = []
            for idx, p in enumerate(perguntas):
                pid = str(p.get("id", p.get("id_pergunta", idx + 1)))
                texto = str(p.get("pergunta", ""))
                if mostrar_completa or len(texto) <= LIMITE:
                    rotulo = f"{pid} — {texto}"
                else:
                    rotulo = f"{pid} — {texto[:LIMITE]}…"
                marcado = st.checkbox(
                    rotulo, key=item_keys[idx], on_change=_on_toggle_item,
                )
                if marcado:
                    selecionadas.append(p)

            st.write(f"**{len(selecionadas)}** pergunta(s) selecionada(s).")

            if st.button("▶️ Executar em lote", type="primary", key="btn_lote"):
                # Guarda a seleção e sinaliza a execução; a renderização
                # (logs + tabela) acontece na seção de acompanhamento abaixo,
                # evitando conteúdo duplicado logo abaixo do botão.
                ss["lote_selecionadas"] = selecionadas
                ss["iniciar_lote"] = True

    # ---- Execução Individual --------------------------------------------
    with sub_ind:
        st.caption(
            "Teste uma pergunta avulsa. O gabarito é opcional; sem ele, o Juiz "
            "avalia com base apenas nas fontes recuperadas."
        )
        pergunta = st.text_area("Digitar pergunta", height=100, key="ind_pergunta")
        gabarito = st.text_area(
            "Digitar gabarito (opcional)", height=100, key="ind_gabarito"
        )
        if st.button("▶️ Executar", type="primary", key="btn_individual"):
            _executar_individual(pergunta, gabarito)

    # ---- Acompanhamento da execução em lote -----------------------------
    # Renderiza a seção UMA única vez, na ordem: Console de Logs -> Tabela.
    # Os placeholders (st.empty) são reutilizados tanto pelo streaming ao vivo
    # quanto pela exibição pós-execução, evitando duplicação.
    iniciar = ss.pop("iniciar_lote", False)

    if iniciar or ss["logs"] or ss["resultados_parciais"]:
        st.divider()
        st.subheader("🖥️ Console de Logs")
        log_box = st.empty()
        progress_box = st.empty()
        st.subheader("📊 Tabela de Resultados")
        tabela_box = st.empty()

        if iniciar:
            # Executa agora, transmitindo os logs/tabela para os placeholders
            # criados acima (na ordem correta).
            _iniciar_benchmarking(
                perguntas=ss.get("lote_selecionadas"),
                gabarito=ss.get("gabarito"),
                log_box=log_box,
                tabela_box=tabela_box,
                progress_box=progress_box,
            )
        else:
            # Exibição pós-execução (reruns seguintes): mostra o último estado.
            _render_logs(log_box, ss["logs"][-200:], altura=400)
            if ss["resultados_parciais"]:
                tabela_box.dataframe(
                    _df_parcial(ss["resultados_parciais"]),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Detalhamento da avaliação do Juiz (notas intermediárias)
# ---------------------------------------------------------------------------

# Critérios de nota (0–5) exibidos como barras de progresso.
_CRITERIOS_NOTA = [
    ("nota_fidelidade", "Fidelidade", "Fidelidade às fontes / ausência de invenção"),
    ("nota_precisao_gabarito", "Precisão vs. Gabarito", "Proximidade com o gabarito oficial"),
    ("nota_completude", "Completude", "Cobertura dos pontos essenciais"),
]


def _painel_validacao_processos(procs: list):
    """Exibe a validação factual de processos citados (DataJud)."""
    if not procs:
        return
    st.caption("⚖️ Processos citados (validação DataJud):")
    for p in procs:
        num = p.get("numero", "")
        if p.get("erro"):
            st.markdown(f"- `{num}` — ⚪ {p['erro']}")
        elif not p.get("existe"):
            st.markdown(f"- `{num}` — 🔴 **Não encontrado** (possível alucinação)")
        else:
            assuntos = ", ".join(p.get("assuntos", [])) or "(sem assunto)"
            if p.get("compativel_tema"):
                st.markdown(f"- `{num}` — 🟢 Existe | assunto: *{assuntos}* | tema compatível")
            else:
                st.markdown(f"- `{num}` — 🟠 Existe | assunto: *{assuntos}* | tema **incompatível**")


def _painel_avaliacao(
    titulo: str, aval: dict, procs: list | None = None, resposta: str | None = None,
    mostrar_resposta: bool = True,
):
    """Renderiza a resposta, as notas intermediárias e os sinais de uma abordagem.

    `mostrar_resposta=False` omite o expander com o texto da resposta (útil quando
    a resposta já é exibida em outro lugar, p.ex. na comparação lado a lado).
    """
    st.markdown(f"**{titulo}**")

    # Texto da resposta gerada pela LLM (opcional).
    if mostrar_resposta:
        with st.expander("💬 Ver resposta gerada", expanded=False):
            st.write(resposta or "(sem resposta)")

    if not aval:
        st.info("Sem avaliação disponível.")
        return

    # Notas 0–5 como barras de progresso.
    for campo, rotulo, ajuda in _CRITERIOS_NOTA:
        nota = aval.get(campo)
        nota = 0 if nota is None else nota
        st.caption(f"{rotulo}: **{nota}/5**", help=ajuda)
        st.progress(min(max(int(nota), 0), 5) / 5)

    # Sentimento processual identificado e sua aderência ao gabarito.
    sent = aval.get("sentimento_identificado", "nao_aplicavel")
    aderente = aval.get("sentimento_aderente_gabarito")
    c1, c2 = st.columns(2)
    c1.markdown(f"🧭 **Sentimento:** {sent}")
    if aderente is True:
        c2.markdown("🟢 **Sentimento aderente ao gabarito**")
    elif aderente is False:
        c2.markdown("🔴 **Sentimento divergente do gabarito**")
    else:
        c2.markdown("⚪ **Sentimento n/a**")

    # Julgados: solicitados x apresentados x verificáveis.
    solicit = aval.get("julgados_solicitados") or 0
    apres = aval.get("julgados_apresentados") or 0
    verif = aval.get("julgados_verificaveis") or 0
    if solicit or apres:
        st.caption(
            f"⚖️ Julgados: {apres}/{solicit} apresentados "
            f"({verif} verificáveis)"
        )

    # Erros críticos (violações do campo Inaceitável).
    erros = aval.get("erros_criticos") or []
    if isinstance(erros, list) and erros:
        st.markdown(f"🔴 **{len(erros)} erro(s) crítico(s):**")
        for e in erros:
            st.markdown(f"- {e}")
    else:
        st.markdown("🟢 **Sem erros críticos**")

    # Justificativa do juiz.
    just = aval.get("justificativa_juiz")
    if just:
        st.caption("📝 Justificativa do Juiz:")
        st.info(just)

    # Validação factual de processos citados (DataJud).
    if procs:
        _painel_validacao_processos(procs)


def _secao_detalhamento_juiz(relatorio: dict):
    """Seção que expõe como o Juiz formou a decisão para cada pergunta."""
    resultados = relatorio.get("resultados_detalhados", [])
    if not resultados:
        return

    st.subheader("🧑‍⚖️ Detalhamento da Avaliação do modelo LLM Juiz")
    st.caption(
        "Compare lado a lado o gabarito e as respostas (Padrão e RAG) e veja as "
        "notas e os sinais que compõem a decisão do Juiz para cada pergunta."
    )

    opcoes = {
        f"{r.get('id_pergunta')} — {str(r.get('pergunta', ''))[:60]}": r
        for r in resultados
    }
    escolha = st.selectbox("Pergunta", list(opcoes.keys()))
    r = opcoes[escolha]

    # Pergunta em largura total (contexto da comparação).
    with st.expander("📄 Ver enunciado da pergunta", expanded=False):
        st.write(r.get("pergunta", ""))

    # Comparação lado a lado: Gabarito x Resposta Padrão x Resposta RAG.
    st.markdown("**🔎 Comparação: Gabarito × LLM Padrão × LLM com RAG**")
    # Força a cor preta no texto das caixas (text_area desabilitado fica cinza
    # claro por padrão, dificultando a leitura).
    st.markdown(
        """
        <style>
        div[data-testid="stTextArea"] textarea:disabled {
            -webkit-text-fill-color: #000000 !important;
            color: #000000 !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cg, cp, cr = st.columns(3)
    with cg:
        st.markdown("📘 **Gabarito**")
        st.text_area(
            "Gabarito",
            value=r.get("gabarito_oficial", "") or "(sem gabarito)",
            height=320, disabled=True, label_visibility="collapsed",
            key=f"cmp_gab_{r.get('id_pergunta')}",
        )
    with cp:
        st.markdown("🔵 **Resposta LLM Padrão**")
        st.text_area(
            "Resposta Padrão",
            value=r.get("resposta_llm_padrao", "") or "(sem resposta)",
            height=320, disabled=True, label_visibility="collapsed",
            key=f"cmp_padrao_{r.get('id_pergunta')}",
        )
    with cr:
        st.markdown("🟣 **Resposta LLM com RAG**")
        st.text_area(
            "Resposta RAG",
            value=r.get("resposta_llm_rag", "") or "(sem resposta)",
            height=320, disabled=True, label_visibility="collapsed",
            key=f"cmp_rag_{r.get('id_pergunta')}",
        )

    st.divider()
    st.markdown("**⚖️ Avaliação do Juiz (notas e sinais)**")

    col_p, col_r = st.columns(2)
    with col_p:
        _painel_avaliacao(
            "🔵 LLM Padrão",
            r.get("avaliacao_llm_padrao", {}),
            r.get("validacao_processos_padrao", []),
            r.get("resposta_llm_padrao", ""),
            mostrar_resposta=False,
        )
    with col_r:
        _painel_avaliacao(
            "🟣 LLM com RAG",
            r.get("avaliacao_llm_rag", {}),
            r.get("validacao_processos_rag", []),
            r.get("resposta_llm_rag", ""),
            mostrar_resposta=False,
        )


# ---------------------------------------------------------------------------
# Aba 3 — Dashboard
# ---------------------------------------------------------------------------

def _carregar_relatorio_bytes(conteudo: bytes) -> dict:
    """
    Reconstrói o dict de relatório a partir dos bytes de um JSON baixado
    anteriormente (botão 'Baixar JSON' da aba Resultados).

    Valida a estrutura mínima esperada pelo dashboard: um objeto com
    'execucao_metadata' (contendo 'metricas_qualidade_global') e a lista
    'resultados_detalhados'. Levanta ValueError com mensagem amigável se o
    arquivo não for um relatório válido.
    """
    try:
        dados = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Arquivo não é um JSON válido: {exc}") from exc

    if not isinstance(dados, dict):
        raise ValueError("O JSON não é um objeto de relatório.")

    meta = dados.get("execucao_metadata")
    if not isinstance(meta, dict) or "metricas_qualidade_global" not in meta:
        raise ValueError(
            "JSON sem 'execucao_metadata.metricas_qualidade_global'. "
            "Envie o arquivo gerado pelo botão 'Baixar JSON' desta tela."
        )
    if "resultados_detalhados" not in dados:
        raise ValueError(
            "JSON sem 'resultados_detalhados'. "
            "Envie o relatório completo gerado por esta aplicação."
        )
    return dados


def _painel_importar_resultados():
    """
    Permite recarregar um relatório salvo (JSON baixado) para revisualizar os
    resultados de uma execução anterior, mesmo após reiniciar a aplicação.

    Lista também os relatórios já presentes em outputs/ para carga com 1 clique.
    """
    ss = st.session_state
    cfg: Config = ss["config"]

    with st.expander("📂 Carregar resultados salvos (de uma execução anterior)",
                     expanded=not ss.get("relatorio")):
        st.caption(
            "Recupere uma análise já processada sem reexecutar. Envie o arquivo "
            "JSON baixado nesta tela (botão 'Baixar JSON') ou escolha um dos "
            "relatórios salvos automaticamente em outputs/."
        )

        up = st.file_uploader(
            "Enviar relatório (.json)", type=["json"], key="up_relatorio"
        )
        if up is not None:
            try:
                relatorio = _carregar_relatorio_bytes(up.getvalue())
            except ValueError as exc:
                st.error(f"Falha ao carregar '{up.name}': {exc}")
            else:
                # Não usamos st.rerun() aqui: o uploader mantém o arquivo entre
                # reruns, então um rerun reentraria neste bloco e interromperia a
                # renderização ANTES do dashboard (tela em branco). Basta setar o
                # relatório na sessão e deixar o fluxo seguir para o dashboard,
                # que é desenhado logo abaixo nesta mesma execução.
                ss["relatorio"] = relatorio
                st.success(f"✅ Relatório '{up.name}' carregado.")

        # Relatórios salvos automaticamente em outputs/ (relatorio_*.json).
        diretorio_saida = Path(cfg.dados.get("diretorio_saida", "outputs/"))
        salvos = sorted(
            diretorio_saida.glob("relatorio_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if diretorio_saida.exists() else []

        if salvos:
            import datetime as _dt

            rotulos = {
                f"{p.name}  "
                f"({_dt.datetime.fromtimestamp(p.stat().st_mtime).strftime('%d/%m/%Y %H:%M')})": p
                for p in salvos
            }
            col_sel, col_btn = st.columns([0.7, 0.3])
            with col_sel:
                escolha = st.selectbox(
                    "Ou carregue um relatório salvo em outputs/",
                    list(rotulos.keys()),
                    key="sel_relatorio_salvo",
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📥 Carregar", use_container_width=True,
                             key="btn_carregar_salvo"):
                    caminho = rotulos[escolha]
                    try:
                        relatorio = _carregar_relatorio_bytes(caminho.read_bytes())
                    except ValueError as exc:
                        st.error(f"Falha ao carregar '{caminho.name}': {exc}")
                    else:
                        ss["relatorio"] = relatorio
                        st.success(f"✅ Relatório '{caminho.name}' carregado.")
        else:
            st.caption("Nenhum relatório salvo encontrado em outputs/.")


def _aba_dashboard():
    st.header("📈 Resultados")
    ss = st.session_state

    # Permite recarregar uma execução anterior (JSON baixado ou de outputs/),
    # para revisualizar análises sem reprocessar após reiniciar o sistema.
    _painel_importar_resultados()

    relatorio = ss.get("relatorio")
    if not relatorio:
        st.info(
            "Execute um benchmarking para ver os resultados aqui, ou carregue um "
            "relatório salvo acima."
        )
        return

    meta = relatorio["execucao_metadata"]
    m = meta["metricas_qualidade_global"]

    # Parecer final consolidado da LLM Juiz
    parecer = meta.get("parecer_final_juiz") or {}
    if parecer.get("parecer_texto"):
        vencedora = parecer.get("abordagem_vencedora", "Indefinido")
        st.subheader("⚖️ Parecer Final da LLM Juiz")
        st.success(f"**Abordagem com melhor desempenho: {vencedora}**")
        with st.expander("Ver parecer descritivo completo", expanded=True):
            st.markdown(parecer["parecer_texto"])
        st.divider()

    st.subheader("🏆 KPIs Globais")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fidelidade (Padrão)", m["nota_fidelidade_media_padrao"])
    c1.metric("Fidelidade (RAG)", m["nota_fidelidade_media_rag"])
    c2.metric("Precisão vs Gabarito (Padrão)", m["nota_precisao_media_padrao"])
    c2.metric("Precisão vs Gabarito (RAG)", m["nota_precisao_media_rag"])
    c3.metric("Erros críticos % (Padrão)", m["taxa_erros_criticos_padrao_pct"])
    c3.metric("Erros críticos % (RAG)", m["taxa_erros_criticos_rag_pct"])

    st.subheader("⏱️ Performance Temporal")
    t1, t2, t3 = st.columns(3)
    t1.metric("Tempo médio Padrão (s)", m["tempo_medio_llm_padrao_s"])
    t2.metric("Tempo médio RAG (s)", m["tempo_medio_llm_rag_s"])
    t3.metric("Overhead RAG (s)", m["overhead_medio_rag_s"], f"{m['overhead_medio_rag_pct']}%")

    df = pd.DataFrame(utils.relatorio_para_linhas_csv(relatorio))

    st.subheader("📊 Comparativo de Qualidade (médias)")
    df_q = pd.DataFrame({
        "Abordagem": ["LLM Padrão", "LLM RAG"],
        "Fidelidade": [m["nota_fidelidade_media_padrao"], m["nota_fidelidade_media_rag"]],
        "Precisão": [m["nota_precisao_media_padrao"], m["nota_precisao_media_rag"]],
        "Completude": [m["nota_completude_media_padrao"], m["nota_completude_media_rag"]],
    })
    df_q_long = df_q.melt(
        id_vars="Abordagem", var_name="Critério", value_name="Nota"
    )
    fig_q = px.bar(
        df_q_long,
        x="Abordagem",
        y="Nota",
        color="Critério",
        barmode="group",
        text="Nota",
        color_discrete_map={
            "Fidelidade": "#1f77b4",
            "Precisão": "#7fb3e0",
            "Completude": "#0b2545",
        },
    )
    fig_q.update_xaxes(tickangle=0, title_text="")
    fig_q.update_layout(legend_title_text="Critério")
    st.plotly_chart(fig_q, use_container_width=True)

    st.subheader("📈 Performance Temporal por Pergunta")
    df_t = df[["id", "tempo_padrao_s", "tempo_total_rag_s"]].copy()
    df_t.columns = ["Pergunta", "LLM Padrão (s)", "LLM RAG (s)"]
    df_t_long = df_t.melt(
        id_vars="Pergunta", var_name="Abordagem", value_name="Tempo (s)"
    )
    fig_t = px.bar(
        df_t_long,
        x="Pergunta",
        y="Tempo (s)",
        color="Abordagem",
        barmode="group",
        color_discrete_map={
            "LLM Padrão (s)": "#1f77b4",
            "LLM RAG (s)": "#0b2545",
        },
    )
    fig_t.update_xaxes(tickangle=0, title_text="")
    st.plotly_chart(fig_t, use_container_width=True)

    _secao_detalhamento_juiz(relatorio)

    st.subheader("🔎 Tabela Detalhada")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        so_aluc = st.checkbox("Só casos com erro crítico (Padrão)")
    with col_f2:
        so_rag_vence = st.checkbox("Só quando RAG > Padrão (precisão)")

    df_view = df.copy()
    if so_aluc:
        df_view = df_view[
            df_view["erros_criticos_padrao"].fillna("").astype(str).str.len() > 0
        ]
    if so_rag_vence:
        df_view = df_view[df_view["nota_precisao_rag"] > df_view["nota_precisao_padrao"]]

    # Reordena as colunas para exibir cada critério em pares
    # (LLM Padrão seguido de LLM RAG), um após o outro.
    ordem_pares = [
        "id", "pergunta", "gabarito",
        "resposta_padrao", "resposta_rag",
        "tempo_padrao_s", "tempo_total_rag_s",
        "nota_fidelidade_padrao", "nota_fidelidade_rag",
        "nota_precisao_padrao", "nota_precisao_rag",
        "nota_completude_padrao", "nota_completude_rag",
        "sentimento_padrao", "sentimento_rag",
        "sentimento_aderente_padrao", "sentimento_aderente_rag",
        "julgados_solicitados",
        "julgados_apresentados_padrao", "julgados_apresentados_rag",
        "julgados_verificaveis_padrao", "julgados_verificaveis_rag",
        "erros_criticos_padrao", "erros_criticos_rag",
        "justificativa_padrao", "justificativa_rag",
        "tempo_retrieval_rag_s", "tempo_geracao_rag_s", "tempo_juiz_s",
    ]
    colunas = [c for c in ordem_pares if c in df_view.columns]
    colunas += [c for c in df_view.columns if c not in colunas]
    df_view = df_view[colunas]
    st.dataframe(df_view, use_container_width=True)

    st.subheader("💾 Exportação")
    col_j, col_c = st.columns(2)
    with col_j:
        st.download_button(
            "⬇️ Baixar JSON",
            data=json.dumps(relatorio, ensure_ascii=False, indent=2),
            file_name="relatorio_llm_tester.json",
            mime="application/json",
        )
    with col_c:
        st.download_button(
            "⬇️ Baixar CSV",
            data=df.to_csv(index=False),
            file_name="resultados_llm_tester.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Aba: Informações do RAG
# ---------------------------------------------------------------------------

def _abrir_collection_chroma(cfg: Config):
    """
    Abre a collection do ChromaDB em modo leitura, com telemetria desativada.
    Retorna (collection, erro). Em caso de falha, collection é None e erro traz
    a mensagem. Não levanta exceção para não quebrar a renderização da aba.
    """
    import os

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")

    params = cfg.parametros
    persist_dir = params.get("persist_directory", "data/chroma_db")
    collection_name = params.get("collection_name", "pensao_por_morte")

    if not Path(persist_dir).exists():
        return None, f"Diretório do ChromaDB não encontrado: {persist_dir}"

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(collection_name)
        return col, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _listar_arquivos_base(cfg: Config) -> list[str]:
    """Lista os arquivos indexáveis (PDF/TXT/CSV) da base jurídica."""
    base = Path(cfg.dados.get("base_juridica", "data/base_juridica/"))
    if not base.exists():
        return []
    arquivos = (
        list(base.glob("**/*.pdf"))
        + list(base.glob("**/*.txt"))
        + list(base.glob("**/*.csv"))
    )
    return sorted(a.name for a in arquivos)


def _disparar_reindexacao(arquivos: list[str] | str | None = None,
                          zerar: bool = False) -> None:
    """
    Dispara o reindexar.py como processo SEPARADO (não trava a UI).

    - arquivos None/vazio: reindexa TODA a base do zero.
    - arquivos com itens: indexa só esses arquivos (incremental).
    - arquivos + zerar=True: zera a base e reindexa só os selecionados.

    O progresso vai para _reindex_log.txt, exibido na própria aba.
    """
    import subprocess
    import os as _os

    # Normaliza para lista.
    if arquivos is None:
        lista = []
    elif isinstance(arquivos, str):
        lista = [arquivos]
    else:
        lista = list(arquivos)

    # Usa o mesmo interpretador Python que roda a UI.
    py = sys.executable
    cmd = [py, "reindexar.py"]
    if lista:
        cmd += ["--arquivos", *lista]
    if zerar:
        cmd += ["--zerar"]

    env = dict(_os.environ)
    env["ANONYMIZED_TELEMETRY"] = "False"
    env["CHROMA_TELEMETRY_IMPL"] = "none"

    # Popen não bloqueia: a indexação roda em background, a UI segue livre.
    subprocess.Popen(cmd, env=env, cwd=_os.getcwd())


def _secao_reindexacao(cfg: Config, contexto: str = "rag"):
    """
    Renderiza os controles de reindexação. `contexto` compõe keys únicas para
    os widgets, permitindo reutilizar esta seção em mais de uma aba sem colidir
    IDs (StreamlitDuplicateElementId).
    """
    st.subheader("🔁 Reindexação")
    st.caption(
        "Selecione os arquivos e clique em Indexar. A indexação roda em segundo "
        "plano; acompanhe o progresso abaixo. Deixe a lista vazia para reindexar "
        "toda a base."
    )

    arquivos = _listar_arquivos_base(cfg)

    # Multisseleção de arquivos. A seleção fica em session_state (com key), o que
    # também permite ao benchmark ('Reindexar do zero ao executar') saber quais
    # arquivos usar.
    selecao = st.multiselect(
        "Escopo da reindexação (selecione um ou mais arquivos)",
        options=arquivos,
        help="Escolha os arquivos a indexar/atualizar. Se nenhum for "
             "selecionado, o botão reindexa TODA a base do zero.",
        key=f"reindex_selecao_{contexto}",
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📄 Indexar selecionados", type="primary",
                     use_container_width=True, disabled=not selecao,
                     key=f"reindex_sel_{contexto}"):
            _disparar_reindexacao(selecao, zerar=False)
            st.session_state["_reindex_disparada"] = True
            st.success(
                f"Indexação de {len(selecao)} arquivo(s) iniciada em segundo plano."
            )
    with col_b2:
        confirmar_key = f"reindex_tudo_confirmar_{contexto}"
        if st.button("♻️ Reindexar toda a base", use_container_width=True,
                     key=f"reindex_tudo_{contexto}"):
            # Não dispara direto: pede confirmação explícita primeiro.
            st.session_state[confirmar_key] = True

    # Caixa de confirmação para a reindexação TOTAL (ação destrutiva: zera e
    # recria toda a base vetorial). Só o clique em "Sim" inicia o processo.
    if st.session_state.get(confirmar_key):
        st.warning(
            "⚠️ **Reindexar TODA a base** vai APAGAR a base vetorial atual e "
            "recriá-la do zero a partir de todos os arquivos da base jurídica. "
            "Os chunks já indexados serão perdidos e o processo pode levar "
            "bastante tempo. Deseja mesmo continuar?"
        )
        col_sim, col_nao = st.columns(2)
        with col_sim:
            if st.button("✅ Sim, reindexar tudo", type="primary",
                         use_container_width=True,
                         key=f"reindex_tudo_sim_{contexto}"):
                st.session_state[confirmar_key] = False
                _disparar_reindexacao(None)
                st.session_state["_reindex_disparada"] = True
                st.warning(
                    "Reindexação TOTAL (todos os arquivos) iniciada em segundo "
                    "plano (pode levar bastante tempo)."
                )
        with col_nao:
            if st.button("❌ Não, cancelar", use_container_width=True,
                         key=f"reindex_tudo_nao_{contexto}"):
                st.session_state[confirmar_key] = False
                st.info("Reindexação total cancelada.")

    # Progresso (lido do arquivo de log gerado pelo reindexar.py)
    log_path = Path("_reindex_log.txt")
    col_r1, col_r2 = st.columns([0.3, 0.7])
    with col_r1:
        atualizar = st.button("🔄 Atualizar progresso",
                              key=f"reindex_atualizar_{contexto}")
    with col_r2:
        if log_path.exists():
            import datetime as _dt
            mtime = _dt.datetime.fromtimestamp(log_path.stat().st_mtime)
            st.caption(f"Log atualizado às {mtime.strftime('%H:%M:%S')}")

    if log_path.exists():
        try:
            linhas = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            # Mostra as últimas linhas (progresso mais recente no fim).
            ultimas = linhas[-25:] if len(linhas) > 25 else linhas
            concluido = any("[FIM]" in ln for ln in linhas)
            erro = any("[ERRO]" in ln for ln in linhas)
            if concluido:
                st.success("✅ Indexação concluída. Atualize a página para ver os novos números.")
            elif erro:
                st.error("❌ A indexação terminou com erro. Veja o log abaixo.")
            _render_logs(st.empty(), ultimas, altura=260)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(Não foi possível ler o log: {exc})")
    else:
        st.caption("Nenhuma reindexação registrada ainda.")


def _aba_rag():
    st.header("🔎 Informações do RAG")
    st.caption(
        "Parâmetros do pipeline de Retrieval-Augmented Generation e o estado "
        "atual da base vetorial (ChromaDB)."
    )

    cfg: Config = st.session_state["config"]
    p = cfg.parametros

    # Helper: renderiza um par rótulo/valor em fonte compacta.
    def _item(rotulo: str, valor) -> str:
        return (
            f"<div style='margin-bottom:8px;'>"
            f"<span style='font-size:0.78em; color:#6b7280;'>{rotulo}</span><br>"
            f"<span style='font-size:1.0em; font-weight:600; color:#0b2545;'>{valor}</span>"
            f"</div>"
        )

    # --- Parâmetros configurados ------------------------------------------
    st.subheader("⚙️ Parâmetros configurados")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**✂️ Chunking**")
        st.markdown(
            _item("Chunk size (chars)", p.get("chunk_size", "—"))
            + _item("Overlap (chars)", p.get("chunk_overlap", "—"))
            + _item("Batch de indexação", p.get("index_batch_size", "—")),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**🧬 Embeddings**")
        st.markdown(
            _item("Modelo", p.get("embedding_model", "—")),
            unsafe_allow_html=True,
        )
        st.caption(
            "Modelo de embeddings executado localmente (CPU) para vetorizar "
            "cada chunk e as consultas."
        )
    with col3:
        st.markdown("**🎯 Recuperação (retrieval)**")
        busca = str(p.get("search_type", "—")).upper()
        html = _item("Top-k", p.get("top_k", "—")) + _item("Tipo de busca", busca)
        if str(p.get("search_type")) == "mmr":
            html += _item("MMR lambda", p.get("mmr_lambda", "—"))
        st.markdown(html, unsafe_allow_html=True)

    st.divider()

    # --- Estado da base vetorial (ChromaDB) --------------------------------
    st.subheader("🗄️ Base vetorial (ChromaDB)")

    ce1, ce2 = st.columns(2)
    ce1.markdown(f"**Collection:** `{p.get('collection_name', '—')}`")
    ce2.markdown(f"**Diretório:** `{p.get('persist_directory', '—')}`")

    col, erro = _abrir_collection_chroma(cfg)
    if erro:
        st.warning(
            f"Não foi possível ler a base vetorial. {erro}\n\n"
            "Se você ainda não indexou, rode a indexação primeiro."
        )
        return

    try:
        total = col.count()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao contar chunks: {exc}")
        return

    st.metric("📦 Total de chunks indexados", f"{total:,}".replace(",", "."))

    if total == 0:
        st.info("A base está vazia. Rode a indexação para populá-la.")
        return

    # --- Distribuição por fonte (arquivo de origem) ------------------------
    st.markdown("**📚 Fontes indexadas (chunks por arquivo)**")
    try:
        # Amostra os metadados para agregar por fonte. Para bases grandes,
        # limita a amostra para não pesar na UI.
        # Busca TODOS os metadados (sem amostrar) para não omitir arquivos.
        # Só metadados é leve — o Chroma agrega dezenas de milhares sem problema.
        dados = col.get(include=["metadatas"])
        metas = dados.get("metadatas", []) or []
        contagem: dict[str, int] = {}
        for meta in metas:
            origem = (meta or {}).get("source") or (meta or {}).get("fonte") or "(desconhecida)"
            origem = Path(str(origem)).name  # só o nome do arquivo
            contagem[origem] = contagem.get(origem, 0) + 1

        if contagem:
            df_fontes = pd.DataFrame(
                sorted(contagem.items(), key=lambda kv: kv[1], reverse=True),
                columns=["Arquivo", "Chunks"],
            )
            st.caption(
                f"{len(df_fontes)} arquivo(s) — {len(metas):,} chunks no total.".replace(",", ".")
            )
            st.dataframe(df_fontes, use_container_width=True, hide_index=True)

            fig = px.bar(
                df_fontes.head(20),
                x="Chunks",
                y="Arquivo",
                orientation="h",
                text="Chunks",
                color_discrete_sequence=["#2f5fa8"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            fig.update_yaxes(title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Os chunks não possuem metadado de fonte para agregar.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Não foi possível agregar por fonte: {exc}")

    st.divider()

    # --- Explorador de busca semântica -------------------------------------
    st.subheader("🔍 Testar recuperação (busca semântica)")
    st.caption(
        "Digite uma consulta e veja quais trechos o RAG recuperaria. Usa o "
        "mesmo modelo de embeddings e a mesma collection da execução."
    )

    consulta = st.text_input(
        "Consulta de teste",
        placeholder="Ex.: requisitos para concessão de pensão por morte",
    )
    n_result = st.slider("Nº de trechos a recuperar", 1, 15, p.get("top_k", 8))

    if st.button("🔎 Buscar", type="primary") and consulta.strip():
        with st.spinner("Recuperando trechos..."):
            try:
                res = col.query(query_texts=[consulta], n_results=n_result)
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]

                if not docs:
                    st.info("Nenhum trecho recuperado.")
                else:
                    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                        origem = Path(str((meta or {}).get("source", "?"))).name
                        with st.expander(
                            f"#{i} — {origem}  (distância: {dist:.4f})",
                            expanded=(i <= 3),
                        ):
                            st.write(doc or "")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro na busca: {exc}")


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def _encerrar_app():
    """Encerra o processo do servidor Streamlit."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGTERM)


def run():
    st.set_page_config(page_title="llm-tester", page_icon="🏛️", layout="wide")
    _init_state()

    # Estiliza o botão primário ("Iniciar Execução") com azul escuro.
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #2f5fa8;
            border-color: #2f5fa8;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #3a70c2;
            border-color: #3a70c2;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:active,
        div.stButton > button[kind="primary"]:focus {
            background-color: #285292;
            border-color: #285292;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_titulo, col_sair = st.columns([0.85, 0.15])
    with col_titulo:
        st.markdown(
            "<h1>🏛️ LLM Padrão x LLM com RAG "
            "<span style='font-size:0.5em; font-weight:600;'>"
            "(Retrieval-Augmented Generation)</span></h1>",
            unsafe_allow_html=True,
        )
        _tema = st.session_state["config"].tema
        _area = _tema.get("area", "Direito Previdenciário")
        _nome_tema = _tema.get("nome", "Pensão por Morte")
        st.caption(f"Domínio: {_area} — {_nome_tema} | Provider: Groq API")
    with col_sair:
        if st.session_state.get("_confirmar_saida"):
            st.warning("Encerrar a aplicação?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Sim", use_container_width=True):
                st.session_state["_encerrado"] = True
                st.rerun()
            if c2.button("❌ Não", use_container_width=True):
                st.session_state["_confirmar_saida"] = False
                st.rerun()
        else:
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state["_confirmar_saida"] = True
                st.rerun()

    if st.session_state.get("_encerrado"):
        st.success("✅ Aplicação encerrada. Você já pode fechar esta aba do navegador.")
        _encerrar_app()
        st.stop()

    aba1, aba2, aba3, aba4, aba5 = st.tabs([
        "⚙️ Configurações", "🗂️ Dados", "▶️ Execução", "📈 Resultados", "🔎 RAG",
    ])
    with aba1:
        _aba_configuracao()
    with aba2:
        _aba_dados_teste()
    with aba3:
        _aba_benchmarking()
    with aba4:
        _aba_dashboard()
    with aba5:
        _aba_rag()
