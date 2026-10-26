"""
Exportação dos resultados do benchmarking para um arquivo Excel (.xlsx)
otimizado para leitura no Power BI.

Gera UM arquivo com três abas (tabelas), no formato "longo" (tidy), onde a
abordagem (Padrão/RAG) é uma COLUNA — e não um sufixo de nome de coluna. Isso
permite usar `abordagem` como eixo/legenda no Power BI sem medidas DAX extras.

Abas:
  - resultados: uma linha por (execução × pergunta × abordagem). Notas, tempos,
    alucinação quantificada (erros críticos, processos citados/confirmados),
    julgados e sentimento.
  - metricas_globais: uma linha por (execução × abordagem), com as médias/taxas
    já calculadas pelo pipeline (KPIs prontos para cartões).
  - textos: uma linha por (execução × pergunta × abordagem) com os textos longos
    (resposta e justificativa) + gabarito, para drill-through. Fica numa aba
    separada para não pesar as tabelas analíticas.

O arquivo é gravado com NOME FIXO (sobrescrevendo), para que, no Power BI, a
fonte seja apontada uma única vez e depois só "Atualizada".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


NOME_ARQUIVO_PADRAO = "powerbi_dashboard.xlsx"


def _carregar_relatorio(caminho: str | Path) -> dict:
    """Lê um relatório JSON do disco."""
    with open(caminho, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _execucao_id(caminho: Path, meta: dict) -> str:
    """
    Identificador estável da execução para o Power BI empilhar várias execuções.
    Usa o nome do arquivo (sem extensão), que já traz o timestamp
    (relatorio_YYYYMMDD_HHMMSS).
    """
    return caminho.stem


def _contar_processos(validacao: list[dict] | None) -> dict[str, int]:
    """
    Resume a validação factual de processos de uma abordagem em contagens:
      citados, confirmados (DataJud ou base CJF), confirmados_base (só base),
      nao_confirmados (possível alucinação).
    """
    procs = validacao or []
    citados = len(procs)
    confirmados = sum(1 for p in procs if p.get("existe") or p.get("na_base_cjf"))
    confirmados_base = sum(
        1 for p in procs if p.get("na_base_cjf") and not p.get("existe")
    )
    nao_confirmados = citados - confirmados
    return {
        "processos_citados": citados,
        "processos_confirmados": confirmados,
        "processos_confirmados_base_cjf": confirmados_base,
        "processos_nao_confirmados": nao_confirmados,
    }


def _qtd_erros(av: dict) -> int:
    erros = av.get("erros_criticos") or []
    return len(erros) if isinstance(erros, list) else (1 if str(erros).strip() else 0)


def _bool_para_int(valor: Any) -> int | None:
    """True/False -> 1/0; None -> None (para o Power BI tratar como vazio)."""
    if valor is True:
        return 1
    if valor is False:
        return 0
    return None


def _chave_exec_abordagem(exec_id: str, abordagem: str) -> str:
    """
    Chave que liga 'metricas_globais' (lado 1) a 'resultados'/'textos' (lado N).
    Granularidade: execução × abordagem.
    """
    return f"{exec_id}|{abordagem}"


def _chave_linha(exec_id: str, id_pergunta: Any, abordagem: str) -> str:
    """
    Chave que liga 'resultados' a 'textos' (relação 1:1).
    Granularidade: execução × pergunta × abordagem.
    """
    return f"{exec_id}|{id_pergunta}|{abordagem}"


def _linhas_resultados(relatorio: dict, exec_id: str, data_hora: str,
                       modelo: str, modelo_juiz: str) -> list[dict]:
    """Monta as linhas da aba 'resultados' (formato longo por abordagem)."""
    linhas: list[dict] = []
    for r in relatorio.get("resultados_detalhados", []):
        pid = r.get("id_pergunta")
        # Dados comuns às duas abordagens.
        base_comum = {
            "execucao_id": exec_id,
            "data_hora": data_hora,
            "modelo_testado": modelo,
            "modelo_juiz": modelo_juiz,
            "id_pergunta": pid,
        }

        # Mapas (abordagem -> fontes de dados daquela abordagem).
        especificos = {
            "Padrao": {
                "aval": r.get("avaliacao_llm_padrao", {}) or {},
                "tempo_total_s": r.get("tempo_execucao_padrao_s"),
                "tempo_retrieval_s": None,
                "tempo_geracao_s": None,
                "validacao": r.get("validacao_processos_padrao"),
            },
            "RAG": {
                "aval": r.get("avaliacao_llm_rag", {}) or {},
                "tempo_total_s": r.get("tempo_total_rag_s"),
                "tempo_retrieval_s": r.get("tempo_retrieval_rag_s"),
                "tempo_geracao_s": r.get("tempo_geracao_rag_s"),
                "validacao": r.get("validacao_processos_rag"),
            },
        }

        for abordagem, dados in especificos.items():
            av = dados["aval"]
            qtd_erros = _qtd_erros(av)
            contagem = _contar_processos(dados["validacao"])
            linha = {
                # Chaves de relacionamento (Power BI).
                "chave_linha": _chave_linha(exec_id, pid, abordagem),
                "chave_exec_abordagem": _chave_exec_abordagem(exec_id, abordagem),
                **base_comum,
                "abordagem": abordagem,
                # Notas (0-5)
                "nota_fidelidade": av.get("nota_fidelidade"),
                "nota_precisao": av.get("nota_precisao_gabarito"),
                "nota_completude": av.get("nota_completude"),
                # Tempos
                "tempo_total_s": dados["tempo_total_s"],
                "tempo_retrieval_s": dados["tempo_retrieval_s"],
                "tempo_geracao_s": dados["tempo_geracao_s"],
                "tempo_avaliacao_juiz_s": r.get("tempo_avaliacao_juiz_s"),
                # Erros críticos / alucinação
                "erros_criticos_qtd": qtd_erros,
                "tem_erro_critico": 1 if qtd_erros > 0 else 0,
                # Julgados
                "julgados_solicitados": av.get("julgados_solicitados"),
                "julgados_apresentados": av.get("julgados_apresentados"),
                "julgados_verificaveis": av.get("julgados_verificaveis"),
                # Sentimento
                "sentimento": av.get("sentimento_identificado"),
                "sentimento_aderente": _bool_para_int(
                    av.get("sentimento_aderente_gabarito")
                ),
                # Validação factual de processos (DataJud + base CJF)
                **contagem,
            }
            linhas.append(linha)
    return linhas


def _linhas_metricas_globais(relatorio: dict, exec_id: str, data_hora: str,
                             modelo: str) -> list[dict]:
    """Monta as linhas da aba 'metricas_globais' (uma por execução × abordagem)."""
    meta = relatorio.get("execucao_metadata", {}) or {}
    m = meta.get("metricas_qualidade_global", {}) or {}
    parecer = meta.get("parecer_final_juiz", {}) or {}
    vencedora = parecer.get("abordagem_vencedora")

    linhas = []
    for abordagem, suf in (("Padrao", "padrao"), ("RAG", "rag")):
        linhas.append({
            # Chave primária desta aba (1 linha por execução × abordagem).
            # Liga-se a resultados/textos (lado N) por este mesmo campo.
            "chave_exec_abordagem": _chave_exec_abordagem(exec_id, abordagem),
            "execucao_id": exec_id,
            "data_hora": data_hora,
            "modelo_testado": modelo,
            "abordagem": abordagem,
            "nota_fidelidade_media": m.get(f"nota_fidelidade_media_{suf}"),
            "nota_precisao_media": m.get(f"nota_precisao_media_{suf}"),
            "nota_completude_media": m.get(f"nota_completude_media_{suf}"),
            "taxa_erros_criticos_pct": m.get(f"taxa_erros_criticos_{suf}_pct"),
            "taxa_sentimento_aderente_pct": m.get(
                f"taxa_sentimento_aderente_{suf}_pct"
            ),
            "tempo_medio_s": m.get(f"tempo_medio_llm_{suf}_s"),
            "overhead_rag_s": m.get("overhead_medio_rag_s") if suf == "rag" else None,
            "overhead_rag_pct": m.get("overhead_medio_rag_pct") if suf == "rag" else None,
            "abordagem_vencedora_execucao": vencedora,
        })
    return linhas


def _linhas_textos(relatorio: dict, exec_id: str) -> list[dict]:
    """Monta as linhas da aba 'textos' (drill-through: respostas/justificativas)."""
    linhas = []
    for r in relatorio.get("resultados_detalhados", []):
        pid = r.get("id_pergunta")
        gabarito = r.get("gabarito_oficial")
        pergunta = r.get("pergunta")
        for abordagem, resp_key, aval_key in (
            ("Padrao", "resposta_llm_padrao", "avaliacao_llm_padrao"),
            ("RAG", "resposta_llm_rag", "avaliacao_llm_rag"),
        ):
            av = r.get(aval_key, {}) or {}
            erros = av.get("erros_criticos") or []
            erros_txt = " | ".join(str(e) for e in erros) if isinstance(erros, list) else str(erros)
            linhas.append({
                # Chaves de relacionamento (Power BI).
                "chave_linha": _chave_linha(exec_id, pid, abordagem),
                "chave_exec_abordagem": _chave_exec_abordagem(exec_id, abordagem),
                "execucao_id": exec_id,
                "id_pergunta": pid,
                "abordagem": abordagem,
                "pergunta": pergunta,
                "gabarito": gabarito,
                "resposta": r.get(resp_key),
                "erros_criticos": erros_txt,
                "justificativa_juiz": av.get("justificativa_juiz"),
            })
    return linhas


def montar_tabelas(
    caminhos_relatorios: Iterable[str | Path],
) -> dict[str, "pd.DataFrame"]:
    """
    Lê os relatórios selecionados e devolve os três DataFrames
    ('resultados', 'metricas_globais', 'textos'), consolidando as execuções.
    """
    res: list[dict] = []
    glob: list[dict] = []
    txt: list[dict] = []

    for caminho in caminhos_relatorios:
        caminho = Path(caminho)
        try:
            relatorio = _carregar_relatorio(caminho)
        except (OSError, json.JSONDecodeError):
            continue
        meta = relatorio.get("execucao_metadata", {}) or {}
        exec_id = _execucao_id(caminho, meta)
        data_hora = meta.get("data_hora", "")
        modelo = meta.get("modelo_testado", "")
        modelo_juiz = meta.get("modelo_juiz", "")

        res.extend(_linhas_resultados(relatorio, exec_id, data_hora, modelo, modelo_juiz))
        glob.extend(_linhas_metricas_globais(relatorio, exec_id, data_hora, modelo))
        txt.extend(_linhas_textos(relatorio, exec_id))

    return {
        "resultados": pd.DataFrame(res),
        "metricas_globais": pd.DataFrame(glob),
        "textos": pd.DataFrame(txt),
    }


def exportar_excel(
    caminhos_relatorios: Iterable[str | Path],
    destino: str | Path,
) -> str:
    """
    Gera o arquivo Excel (.xlsx) com as três abas a partir dos relatórios
    selecionados. Sobrescreve o destino. Retorna o caminho do arquivo.
    """
    tabelas = montar_tabelas(caminhos_relatorios)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        for aba, df in tabelas.items():
            # Excel limita o nome da aba a 31 chars (ok aqui) e não aceita df vazio
            # sem colunas; garantimos ao menos o cabeçalho.
            (df if not df.empty else pd.DataFrame(columns=["(sem dados)"])).to_excel(
                writer, sheet_name=aba, index=False
            )
    return str(destino)
