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
    """Aceita dict com chave 'perguntas' ou lista direta."""
    if isinstance(dados, dict):
        return dados.get("perguntas", [])
    if isinstance(dados, list):
        return dados
    return []


def _normalizar_gabarito(dados: Any) -> list[dict]:
    """Aceita dict com chave 'gabarito' ou lista direta."""
    if isinstance(dados, dict):
        return dados.get("gabarito", [])
    if isinstance(dados, list):
        return dados
    return []


def carregar_perguntas(caminho: str | Path) -> list[dict]:
    """Carrega perguntas de um arquivo .json ou .csv."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {caminho}")

    if caminho.suffix.lower() == ".csv":
        return carregar_csv(caminho)
    with open(caminho, "r", encoding="utf-8") as f:
        return _normalizar_perguntas(json.load(f))


def carregar_gabarito(caminho: str | Path) -> list[dict]:
    """Carrega gabarito de um arquivo .json ou .csv."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de gabarito não encontrado: {caminho}")

    if caminho.suffix.lower() == ".csv":
        return carregar_csv(caminho)
    with open(caminho, "r", encoding="utf-8") as f:
        return _normalizar_gabarito(json.load(f))


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


def carregar_perguntas_bytes(conteudo: bytes, nome_arquivo: str) -> list[dict]:
    """Carrega perguntas a partir de bytes (upload Streamlit)."""
    if nome_arquivo.lower().endswith(".csv"):
        return _ler_csv_texto(conteudo.decode("utf-8"))
    return _normalizar_perguntas(json.loads(conteudo.decode("utf-8")))


def carregar_gabarito_bytes(conteudo: bytes, nome_arquivo: str) -> list[dict]:
    """Carrega gabarito a partir de bytes (upload Streamlit)."""
    if nome_arquivo.lower().endswith(".csv"):
        return _ler_csv_texto(conteudo.decode("utf-8"))
    return _normalizar_gabarito(json.loads(conteudo.decode("utf-8")))


def carregar_csv(caminho: str | Path) -> list[dict]:
    with open(caminho, "r", encoding="utf-8") as f:
        return _ler_csv_texto(f.read())


def _ler_csv_texto(texto: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(texto))
    return [dict(row) for row in reader]


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

    def taxa_alucinacao(abordagem: str) -> float:
        flags = [
            bool(r.get(abordagem, {}).get("alucinacao_detectada"))
            for r in resultados
            if r.get(abordagem)
        ]
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
        "taxa_alucinacao_padrao_pct": taxa_alucinacao("avaliacao_llm_padrao"),
        "taxa_alucinacao_rag_pct": taxa_alucinacao("avaliacao_llm_rag"),
    }


# ---------------------------------------------------------------------------
# Exportação de relatórios
# ---------------------------------------------------------------------------

def montar_relatorio(
    resultados: list[dict],
    modelo_testado: str,
    modelo_juiz: str,
) -> dict[str, Any]:
    """Monta o dicionário final conforme o contrato do JSON de saída."""
    metricas = calcular_metricas_globais(resultados)
    return {
        "execucao_metadata": {
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "modelo_testado": modelo_testado,
            "modelo_juiz": modelo_juiz,
            "total_perguntas": len(resultados),
            "performance_temporal_global": {
                "tempo_medio_llm_padrao_s": metricas.get("tempo_medio_llm_padrao_s", 0.0),
                "tempo_medio_llm_rag_s": metricas.get("tempo_medio_llm_rag_s", 0.0),
                "overhead_medio_rag_s": metricas.get("overhead_medio_rag_s", 0.0),
            },
            "metricas_qualidade_global": metricas,
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
        linhas.append({
            "id": r.get("id_pergunta"),
            "pergunta": r.get("pergunta"),
            "gabarito": r.get("gabarito_oficial"),
            "resposta_padrao": r.get("resposta_llm_padrao"),
            "tempo_padrao_s": r.get("tempo_execucao_padrao_s"),
            "nota_fidelidade_padrao": av_p.get("nota_fidelidade"),
            "nota_precisao_padrao": av_p.get("nota_precisao_gabarito"),
            "nota_completude_padrao": av_p.get("nota_completude"),
            "alucinacao_padrao": av_p.get("alucinacao_detectada"),
            "resposta_rag": r.get("resposta_llm_rag"),
            "tempo_retrieval_rag_s": r.get("tempo_retrieval_rag_s"),
            "tempo_geracao_rag_s": r.get("tempo_geracao_rag_s"),
            "tempo_total_rag_s": r.get("tempo_total_rag_s"),
            "nota_fidelidade_rag": av_r.get("nota_fidelidade"),
            "nota_precisao_rag": av_r.get("nota_precisao_gabarito"),
            "nota_completude_rag": av_r.get("nota_completude"),
            "alucinacao_rag": av_r.get("alucinacao_detectada"),
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
    return str(caminho)
