"""
Utilidades do llm-tester: leitura de datasets (JSON/CSV), rate-limit backoff,
cálculo de métricas agregadas e exportação de relatórios (JSON/CSV).
"""

from __future__ import annotations

import io
import csv
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Leitura de datasets (perguntas e gabarito)
# ---------------------------------------------------------------------------

# Tema/área correntes usados no metadata dos datasets salvos.
# Podem ser sobrescritos em runtime via configurar_tema() a partir do config.
_TEMA_ATUAL = {"tema": "Pensão por Morte", "area": "Direito Previdenciário"}


def configurar_tema(tema: str | None, area: str | None) -> None:
    """Define o tema/área usados no metadata ao salvar perguntas/gabarito."""
    if tema:
        _TEMA_ATUAL["tema"] = tema
    if area:
        _TEMA_ATUAL["area"] = area


def _normalizar_perguntas(dados: Any) -> list[dict]:
    """Aceita dict com chave 'perguntas'/'questoes' ou lista direta."""
    if isinstance(dados, dict):
        itens = dados.get("perguntas") or dados.get("questoes") or []
    elif isinstance(dados, list):
        itens = dados
    else:
        itens = []
    return [_normalizar_item_pergunta(it) for it in itens if isinstance(it, dict)]


def _normalizar_item_pergunta(item: dict) -> dict:
    """Garante id ('Qnn') e texto da pergunta, preservando campos extras."""
    norm = dict(item)
    norm["id"] = _normalizar_id_questao(item.get("id", item.get("id_pergunta")))
    # Aceita variações de nome para o texto da pergunta.
    norm["pergunta"] = (
        item.get("pergunta")
        or item.get("enunciado")
        or item.get("texto")
        or ""
    )
    return norm


def _normalizar_gabarito(dados: Any) -> list[dict]:
    """
    Normaliza o gabarito para o formato interno esperado pelo pipeline/Juiz.

    Aceita três formatos de entrada:
      - dict com chave 'gabarito' (formato interno já normalizado);
      - dict com chave 'questoes' (formato do autor do dataset, com campos ricos
        como resposta_esperada, referencias, pontos_obrigatorios, etc.);
      - lista direta de itens.

    Cada item é normalizado por `_normalizar_item_gabarito` para garantir as
    chaves que o restante do código consome (`id`, `gabarito_oficial`, ...).
    """
    if isinstance(dados, dict):
        itens = dados.get("gabarito") or dados.get("questoes") or []
    elif isinstance(dados, list):
        itens = dados
    else:
        itens = []
    return [_normalizar_item_gabarito(it) for it in itens if isinstance(it, dict)]


def _normalizar_id_questao(valor: Any) -> str:
    """
    Normaliza o id de uma questão para o padrão das perguntas ('Qnn').

    - Numérico (1, 2, 10) ou string numérica ('1') -> 'Q01', 'Q02', 'Q10'.
    - Já no formato 'Q01' ou outra string -> mantém como veio (apenas strip).
    """
    if valor is None:
        return ""
    s = str(valor).strip()
    if s.isdigit():
        return f"Q{int(s):02d}"
    return s


def _normalizar_item_gabarito(item: dict) -> dict:
    """
    Converte um item do gabarito para o formato interno, preservando os campos
    ricos do dataset original (referências, pontos obrigatórios, critérios
    inaceitáveis, etc.) para que o Juiz possa avaliá-los.

    Mapeamentos:
      - id (numérico) -> 'Qnn' para casar com o id das perguntas;
      - 'resposta_esperada' | 'gabarito_oficial' | 'resposta_referencia'
        -> 'gabarito_oficial' (texto principal do gabarito).
    """
    norm = dict(item)  # preserva todos os campos originais

    # id normalizado (casa com perguntas 'Qnn')
    norm["id"] = _normalizar_id_questao(item.get("id", item.get("id_pergunta")))

    # texto principal do gabarito (aceita os vários nomes de campo)
    texto = (
        item.get("gabarito_oficial")
        or item.get("resposta_esperada")
        or item.get("resposta_referencia")
        or ""
    )
    norm["gabarito_oficial"] = texto

    return norm


def carregar_perguntas(caminho: str | Path) -> list[dict]:
    """Carrega perguntas de um arquivo .json ou .csv."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {caminho}")

    if caminho.suffix.lower() == ".csv":
        return carregar_csv(caminho)
    # utf-8-sig remove o BOM se presente (editores podem gravar JSON com BOM).
    with open(caminho, "r", encoding="utf-8-sig") as f:
        return _normalizar_perguntas(json.load(f))


def carregar_gabarito(caminho: str | Path) -> list[dict]:
    """Carrega gabarito de um arquivo .json ou .csv."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de gabarito não encontrado: {caminho}")

    if caminho.suffix.lower() == ".csv":
        return carregar_csv(caminho)
    # utf-8-sig remove o BOM se presente (editores podem gravar JSON com BOM).
    with open(caminho, "r", encoding="utf-8-sig") as f:
        return _normalizar_gabarito(json.load(f))


def perguntas_do_gabarito(gabarito: list[dict]) -> list[dict]:
    """
    Deriva a lista de perguntas a partir do gabarito (fonte ÚNICA).

    Cada item do gabarito já contém o `id` e o texto da `pergunta`; esta função
    extrai esses campos (mais `categoria`, quando presente) no formato esperado
    pelo restante do pipeline. Evita manter um dataset de perguntas separado e
    sujeito a dessincronização com o gabarito.
    """
    perguntas: list[dict] = []
    for item in gabarito:
        if not isinstance(item, dict):
            continue
        pergunta = (item.get("pergunta") or "").strip()
        if not pergunta:
            continue
        reg = {
            "id": item.get("id", item.get("id_pergunta", "")),
            "pergunta": pergunta,
        }
        if item.get("categoria"):
            reg["categoria"] = item["categoria"]
        perguntas.append(reg)
    return perguntas


def _salvar_dataset(
    caminho: str | Path, chave_lista: str, itens: list[dict]
) -> None:
    """
    Salva perguntas/gabarito em JSON preservando o bloco 'metadata' existente
    e atualizando 'total_perguntas'.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict = {}
    if caminho.exists():
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                atual = json.load(f)
            if isinstance(atual, dict):
                metadata = atual.get("metadata", {}) or {}
        except (json.JSONDecodeError, OSError):
            metadata = {}

    if metadata:
        metadata["total_perguntas"] = len(itens)

    saida = {"metadata": metadata, chave_lista: itens} if metadata else {chave_lista: itens}
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)


def salvar_perguntas(itens: list[dict], caminho: str | Path) -> None:
    """Salva a lista de perguntas em JSON, preservando metadata."""
    _salvar_dataset(caminho, "perguntas", itens)


def salvar_gabarito(itens: list[dict], caminho: str | Path) -> None:
    """Salva a lista de gabarito em JSON, preservando metadata."""
    _salvar_dataset(caminho, "gabarito", itens)


def _decodificar_json(conteudo: bytes) -> Any:
    """Decodifica bytes JSON tolerando BOM (utf-8-sig) e UTF-8 puro."""
    try:
        texto = conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = conteudo.decode("utf-8", errors="replace")
    return json.loads(texto)


def carregar_perguntas_bytes(conteudo: bytes, nome_arquivo: str) -> list[dict]:
    """Carrega perguntas a partir de bytes (upload Streamlit): JSON, CSV ou XLSX."""
    nome = nome_arquivo.lower()
    if nome.endswith(".csv"):
        return _ler_csv_texto(conteudo.decode("utf-8-sig"))
    if nome.endswith(".xlsx"):
        return _ler_xlsx_perguntas(conteudo)
    return _normalizar_perguntas(_decodificar_json(conteudo))


def carregar_gabarito_bytes(conteudo: bytes, nome_arquivo: str) -> list[dict]:
    """Carrega gabarito a partir de bytes (upload Streamlit): JSON, CSV ou XLSX."""
    nome = nome_arquivo.lower()
    if nome.endswith(".csv"):
        return _ler_csv_texto(conteudo.decode("utf-8-sig"))
    if nome.endswith(".xlsx"):
        # XLSX de gabarito é tratado como linhas livres (colunas viram campos).
        return _normalizar_gabarito(_ler_xlsx_linhas(conteudo))
    return _normalizar_gabarito(_decodificar_json(conteudo))


def carregar_csv(caminho: str | Path) -> list[dict]:
    with open(caminho, "r", encoding="utf-8") as f:
        return _ler_csv_texto(f.read())


def _ler_csv_texto(texto: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(texto))
    return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Leitura de XLSX (sem dependência externa: lê o pacote OOXML diretamente)
# ---------------------------------------------------------------------------

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _ler_xlsx_linhas(conteudo: bytes) -> list[dict]:
    """
    Lê a primeira planilha de um .xlsx e retorna uma lista de dicts, usando a
    primeira linha como cabeçalho (nomes de coluna). Não depende de openpyxl.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    ns = {"a": _XLSX_NS}

    def _texto_celula(el) -> str:
        return "".join(t.text or "" for t in el.iter(f"{{{_XLSX_NS}}}t"))

    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [_texto_celula(si) for si in root.findall("a:si", ns)]

        # Descobre o caminho da primeira planilha.
        nome_sheet = "xl/worksheets/sheet1.xml"
        candidatos = sorted(
            n for n in z.namelist()
            if n.startswith("xl/worksheets/") and n.endswith(".xml")
        )
        if candidatos:
            nome_sheet = candidatos[0]

        root = ET.fromstring(z.read(nome_sheet))
        sheet_data = root.find("a:sheetData", ns)
        if sheet_data is None:
            return []

        matriz: list[dict[str, str]] = []
        for row in sheet_data.findall("a:row", ns):
            cel: dict[str, str] = {}
            for c in row.findall("a:c", ns):
                ref = c.get("r") or ""
                m = re.match(r"[A-Z]+", ref)
                col = m.group(0) if m else ref
                t = c.get("t")
                if t == "inlineStr":
                    isn = c.find("a:is", ns)
                    val = _texto_celula(isn) if isn is not None else ""
                else:
                    v = c.find("a:v", ns)
                    val = v.text if v is not None else ""
                    if t == "s" and val not in (None, ""):
                        try:
                            val = shared[int(val)]
                        except (ValueError, IndexError):
                            val = ""
                cel[col] = (val or "").strip()
            matriz.append(cel)

    if not matriz:
        return []

    # Primeira linha = cabeçalho (mapeia letra da coluna -> nome do campo).
    cabecalho = {col: (nome or col) for col, nome in matriz[0].items()}
    registros: list[dict] = []
    for linha in matriz[1:]:
        reg = {cabecalho.get(col, col): val for col, val in linha.items()}
        if any(v for v in reg.values()):
            registros.append(reg)
    return registros


def _ler_xlsx_perguntas(conteudo: bytes) -> list[dict]:
    """
    Lê um .xlsx de perguntas (colunas tipo 'Sequência'/'Pergunta') e devolve
    itens no formato {id, pergunta, categoria}.
    """
    linhas = _ler_xlsx_linhas(conteudo)
    perguntas: list[dict] = []
    for linha in linhas:
        # Localiza as colunas por nome (tolerante a acentos/caixa).
        seq = ""
        texto = ""
        categoria = ""
        for nome, valor in linha.items():
            chave = str(nome).strip().lower()
            if not valor:
                continue
            if "sequ" in chave or chave in ("id", "nº", "no", "numero", "número"):
                seq = seq or valor
            elif "pergunta" in chave or "enunciado" in chave:
                texto = texto or valor
            elif "categoria" in chave:
                categoria = categoria or valor
        if not texto:
            continue
        perguntas.append({
            "id": _normalizar_id_questao(seq),
            "pergunta": texto,
            "categoria": categoria,
        })
    return perguntas


def indexar_por_id(itens: list[dict], chave: str = "id") -> dict[str, dict]:
    """Cria um dict {id: item} para lookup rápido, normalizando o id p/ str."""
    indice = {}
    for item in itens:
        _id = str(item.get(chave, item.get("id_pergunta", "")))
        indice[_id] = item
    return indice


# ---------------------------------------------------------------------------
# Rate limit: exponential backoff para HTTP 429 (Groq free tier)
# ---------------------------------------------------------------------------

def _e_rate_limit(exc: Exception) -> bool:
    """Detecta se a exceção corresponde a rate limit (HTTP 429)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    texto = str(exc).lower()
    return "429" in texto or "rate limit" in texto or "too many requests" in texto


def com_backoff(
    fn: Callable[[], Any],
    max_retries: int = 6,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    on_wait: Callable[[float, int], None] | None = None,
) -> Any:
    """
    Executa `fn` com exponential backoff em caso de rate limit (429).

    IMPORTANTE: o tempo de espera do backoff NÃO deve ser contabilizado como
    latência da LLM. Por isso a cronometragem é feita por quem chama esta
    função em torno da execução real, e o backoff apenas repete a chamada.
    """
    tentativa = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _e_rate_limit(exc) or tentativa >= max_retries:
                raise
            espera = min(base_delay_s * (2 ** tentativa), max_delay_s)
            espera += random.uniform(0, 0.5 * espera)  # jitter
            if on_wait:
                on_wait(espera, tentativa + 1)
            time.sleep(espera)
            tentativa += 1


# ---------------------------------------------------------------------------
# Métricas agregadas
# ---------------------------------------------------------------------------

def _media(valores: Iterable[float]) -> float:
    vals = [v for v in valores if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def calcular_metricas_globais(resultados: list[dict]) -> dict[str, Any]:
    """Calcula KPIs globais de qualidade e performance temporal."""
    if not resultados:
        return {}

    t_padrao = [r.get("tempo_execucao_padrao_s") for r in resultados]
    t_rag = [r.get("tempo_total_rag_s") for r in resultados]

    def notas(abordagem: str, campo: str) -> list[float]:
        out = []
        for r in resultados:
            av = r.get(abordagem, {})
            if av and campo in av:
                out.append(av[campo])
        return out

    def taxa_erros_criticos(abordagem: str) -> float:
        """% de respostas que incorreram em ao menos um erro crítico."""
        flags = []
        for r in resultados:
            av = r.get(abordagem)
            if not av:
                continue
            erros = av.get("erros_criticos") or []
            flags.append(bool(erros) if isinstance(erros, list) else bool(erros))
        return round(100.0 * sum(flags) / len(flags), 2) if flags else 0.0

    def taxa_sentimento_aderente(abordagem: str) -> float:
        """% de respostas cujo sentimento foi aderente ao gabarito."""
        flags = []
        for r in resultados:
            av = r.get(abordagem)
            if not av:
                continue
            aderente = av.get("sentimento_aderente_gabarito")
            if aderente is None:
                continue  # sentimento não aplicável/indefinido não entra na taxa
            flags.append(bool(aderente))
        return round(100.0 * sum(flags) / len(flags), 2) if flags else 0.0

    media_t_padrao = _media(t_padrao)
    media_t_rag = _media(t_rag)

    return {
        "tempo_medio_llm_padrao_s": media_t_padrao,
        "tempo_medio_llm_rag_s": media_t_rag,
        "overhead_medio_rag_s": round(media_t_rag - media_t_padrao, 4),
        "overhead_medio_rag_pct": (
            round(100.0 * (media_t_rag - media_t_padrao) / media_t_padrao, 2)
            if media_t_padrao else 0.0
        ),
        "nota_fidelidade_media_padrao": _media(notas("avaliacao_llm_padrao", "nota_fidelidade")),
        "nota_fidelidade_media_rag": _media(notas("avaliacao_llm_rag", "nota_fidelidade")),
        "nota_precisao_media_padrao": _media(notas("avaliacao_llm_padrao", "nota_precisao_gabarito")),
        "nota_precisao_media_rag": _media(notas("avaliacao_llm_rag", "nota_precisao_gabarito")),
        "nota_completude_media_padrao": _media(notas("avaliacao_llm_padrao", "nota_completude")),
        "nota_completude_media_rag": _media(notas("avaliacao_llm_rag", "nota_completude")),
        "taxa_erros_criticos_padrao_pct": taxa_erros_criticos("avaliacao_llm_padrao"),
        "taxa_erros_criticos_rag_pct": taxa_erros_criticos("avaliacao_llm_rag"),
        "taxa_sentimento_aderente_padrao_pct": taxa_sentimento_aderente("avaliacao_llm_padrao"),
        "taxa_sentimento_aderente_rag_pct": taxa_sentimento_aderente("avaliacao_llm_rag"),
    }


# ---------------------------------------------------------------------------
# Exportação de relatórios
# ---------------------------------------------------------------------------

def montar_relatorio(
    resultados: list[dict],
    modelo_testado: str,
    modelo_juiz: str,
    tempos_indexacao: dict[str, Any] | None = None,
    parecer_final: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta o dicionário final conforme o contrato do JSON de saída."""
    metricas = calcular_metricas_globais(resultados)
    return {
        "execucao_metadata": {
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "modelo_testado": modelo_testado,
            "modelo_juiz": modelo_juiz,
            "total_perguntas": len(resultados),
            "performance_indexacao": tempos_indexacao or {},
            "performance_temporal_global": {
                "tempo_medio_llm_padrao_s": metricas.get("tempo_medio_llm_padrao_s", 0.0),
                "tempo_medio_llm_rag_s": metricas.get("tempo_medio_llm_rag_s", 0.0),
                "overhead_medio_rag_s": metricas.get("overhead_medio_rag_s", 0.0),
            },
            "metricas_qualidade_global": metricas,
            "parecer_final_juiz": parecer_final or {},
        },
        "resultados_detalhados": resultados,
    }


def salvar_json(relatorio: dict, diretorio_saida: str | Path) -> str:
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = Path(diretorio_saida) / f"relatorio_{ts}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    return str(caminho)


def relatorio_para_linhas_csv(relatorio: dict) -> list[dict]:
    """Achata o relatório detalhado em linhas para CSV/DataFrame."""
    linhas = []
    for r in relatorio.get("resultados_detalhados", []):
        av_p = r.get("avaliacao_llm_padrao", {})
        av_r = r.get("avaliacao_llm_rag", {})

        def _erros(av: dict) -> str:
            erros = av.get("erros_criticos") or []
            if isinstance(erros, list):
                return " | ".join(str(e) for e in erros)
            return str(erros)

        linhas.append({
            "id": r.get("id_pergunta"),
            "pergunta": r.get("pergunta"),
            "gabarito": r.get("gabarito_oficial"),
            "resposta_padrao": r.get("resposta_llm_padrao"),
            "tempo_padrao_s": r.get("tempo_execucao_padrao_s"),
            "nota_fidelidade_padrao": av_p.get("nota_fidelidade"),
            "nota_precisao_padrao": av_p.get("nota_precisao_gabarito"),
            "nota_completude_padrao": av_p.get("nota_completude"),
            "sentimento_padrao": av_p.get("sentimento_identificado"),
            "sentimento_aderente_padrao": av_p.get("sentimento_aderente_gabarito"),
            "julgados_apresentados_padrao": av_p.get("julgados_apresentados"),
            "julgados_verificaveis_padrao": av_p.get("julgados_verificaveis"),
            "erros_criticos_padrao": _erros(av_p),
            "resposta_rag": r.get("resposta_llm_rag"),
            "tempo_retrieval_rag_s": r.get("tempo_retrieval_rag_s"),
            "tempo_geracao_rag_s": r.get("tempo_geracao_rag_s"),
            "tempo_total_rag_s": r.get("tempo_total_rag_s"),
            "nota_fidelidade_rag": av_r.get("nota_fidelidade"),
            "nota_precisao_rag": av_r.get("nota_precisao_gabarito"),
            "nota_completude_rag": av_r.get("nota_completude"),
            "sentimento_rag": av_r.get("sentimento_identificado"),
            "sentimento_aderente_rag": av_r.get("sentimento_aderente_gabarito"),
            "julgados_apresentados_rag": av_r.get("julgados_apresentados"),
            "julgados_verificaveis_rag": av_r.get("julgados_verificaveis"),
            "julgados_solicitados": av_r.get("julgados_solicitados")
                or av_p.get("julgados_solicitados"),
            "erros_criticos_rag": _erros(av_r),
            "justificativa_padrao": av_p.get("justificativa_juiz"),
            "justificativa_rag": av_r.get("justificativa_juiz"),
            "tempo_juiz_s": r.get("tempo_avaliacao_juiz_s"),
        })
    return linhas


def salvar_perguntas(perguntas: list[dict], caminho: str | Path) -> str:
    """Salva a lista de perguntas em JSON, preservando o formato com metadata."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "metadata": {
            "tema": _TEMA_ATUAL["tema"],
            "area": _TEMA_ATUAL["area"],
            "descricao": "Dataset de perguntas de entrada para o benchmarking",
            "total_perguntas": len(perguntas),
        },
        "perguntas": perguntas,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return str(caminho)


def salvar_gabarito(gabarito: list[dict], caminho: str | Path) -> str:
    """Salva a lista de gabaritos em JSON, preservando o formato com metadata."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "metadata": {
            "tema": _TEMA_ATUAL["tema"],
            "area": _TEMA_ATUAL["area"],
            "descricao": "Respostas oficiais de referência e critérios de avaliação",
            "total_perguntas": len(gabarito),
        },
        "gabarito": gabarito,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return str(caminho)


def salvar_csv(relatorio: dict, diretorio_saida: str | Path) -> str:
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = Path(diretorio_saida) / f"resultados_{ts}.csv"
    linhas = relatorio_para_linhas_csv(relatorio)
    if not linhas:
        Path(caminho).write_text("", encoding="utf-8")
        return str(caminho)
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)

        # Seção de resumo com o parecer final da LLM Juiz (dado global).
        parecer = (
            relatorio.get("execucao_metadata", {}).get("parecer_final_juiz") or {}
        )
        if parecer.get("parecer_texto"):
            plain = csv.writer(f)
            plain.writerow([])  # linha em branco separando
            plain.writerow(["PARECER FINAL DA LLM JUIZ"])
            plain.writerow(["abordagem_vencedora", parecer.get("abordagem_vencedora", "")])
            # Parecer em texto: uma célula por linha do parecer, para manter
            # legibilidade ao abrir a planilha.
            for linha_txt in parecer["parecer_texto"].splitlines():
                if linha_txt.strip():
                    plain.writerow([linha_txt])
    return str(caminho)
