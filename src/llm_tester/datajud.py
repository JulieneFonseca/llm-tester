"""
Validação factual de processos judiciais citados nas respostas das LLMs.

Fluxo:
  1. Extrai números de processo (padrão CNJ, com ou sem máscara) do texto.
  2. Consulta cada processo na API Pública do DataJud (CNJ), resolvendo o
     endpoint do TRF a partir do próprio número.
  3. Verifica se o processo existe e recupera o campo "assuntos".
  4. Valida (comparação simples por palavra-chave) se o assunto é compatível
     com o tema configurado. As palavras-chave vêm do config (seção 'tema');
     na ausência, usa-se o padrão do módulo (Previdenciário / Pensão por Morte).

Escopo atual: apenas Tribunais Regionais Federais (TRF1..TRF6).

Autenticação: chave pública do DataJud lida do .env (DATAJUD_API_KEY), com
fallback para a chave pública publicada na wiki do CNJ.

Docs: https://datajud-wiki.cnj.jus.br/api-publica/
"""

from __future__ import annotations

import os
import re
import json
import time
import socket
import urllib.request
import urllib.error
from typing import Any

# Chave pública publicada pelo CNJ (fallback caso não haja no .env).
# https://datajud-wiki.cnj.jus.br/api-publica/acesso/
_DATAJUD_API_KEY_FALLBACK = (
    "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
)

_BASE_URL = "https://api-publica.datajud.cnj.jus.br"

# Palavras-chave para validar compatibilidade do assunto com o tema.
_PALAVRAS_TEMA = ("pensao", "pensão", "morte", "previdenci", "previdenciário", "beneficio")

# Regex para o número único CNJ.
#   Com máscara:  NNNNNNN-DD.AAAA.J.TR.OOOO
#   Sem máscara:  20 dígitos consecutivos
_RE_CNJ_MASCARA = re.compile(r"\b(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})\b")
_RE_CNJ_DIGITOS = re.compile(r"\b(\d{20})\b")

# Mapeamento segmento/tribunal -> alias do endpoint (somente Justiça Federal).
# No número CNJ, J=4 indica Justiça Federal; TR=01..06 indica o TRF.
_TRF_ALIASES = {
    "01": "api_publica_trf1",
    "02": "api_publica_trf2",
    "03": "api_publica_trf3",
    "04": "api_publica_trf4",
    "05": "api_publica_trf5",
    "06": "api_publica_trf6",
}


# Cache em memória: {numero_20_digitos: resultado} para evitar reconsultas.
_CACHE: dict[str, dict[str, Any]] = {}


def limpar_cache() -> None:
    """Limpa o cache de consultas ao DataJud."""
    _CACHE.clear()


def _api_key() -> str:
    return os.getenv("DATAJUD_API_KEY") or _DATAJUD_API_KEY_FALLBACK


def _normalizar_numero(numero: str) -> str:
    """Remove máscara, deixando apenas os 20 dígitos."""
    return re.sub(r"\D", "", numero)


def _resolver_endpoint(numero_digitos: str) -> str | None:
    """
    Resolve o alias do endpoint do TRF a partir do número CNJ (20 dígitos).

    Layout CNJ: NNNNNNN DD AAAA J TR OOOO
    posições:   0-6     7-8 9-12 13 14-15 16-19
    Retorna o alias apenas para Justiça Federal (J=4); caso contrário None.
    """
    if len(numero_digitos) != 20:
        return None
    segmento = numero_digitos[13]
    tribunal = numero_digitos[14:16]
    if segmento != "4":
        return None
    return _TRF_ALIASES.get(tribunal)


def extrair_numeros_processo(texto: str) -> list[str]:
    """Extrai números de processo (CNJ) do texto, sem duplicatas."""
    if not texto:
        return []
    encontrados: list[str] = []
    vistos: set[str] = set()
    for regex in (_RE_CNJ_MASCARA, _RE_CNJ_DIGITOS):
        for m in regex.findall(texto):
            digitos = _normalizar_numero(m)
            if len(digitos) == 20 and digitos not in vistos:
                vistos.add(digitos)
                encontrados.append(m)
    return encontrados


def _assunto_compativel(
    assuntos: list[dict], palavras_tema: tuple | list | None = None
) -> bool:
    """Comparação simples por palavra-chave nos nomes dos assuntos."""
    palavras = palavras_tema or _PALAVRAS_TEMA
    palavras = tuple(str(p).lower() for p in palavras)
    for a in assuntos:
        nome = str(a.get("nome", "")).lower()
        if any(chave in nome for chave in palavras):
            return True
    return False


# ---------------------------------------------------------------------------
# Índice local dos números de processo presentes na base jurídica (CSV CJF)
# ---------------------------------------------------------------------------
#
# Serve como "segunda fonte de verdade" para o RAG: quando o DataJud não
# encontra um processo, verificamos se o número consta nos arquivos da base
# (CJF*.csv). Se constar, o processo NÃO foi inventado pela LLM — ele foi
# extraído da base indexada, apenas não está (ou não foi localizado) na API
# pública do DataJud. Isso evita falsos positivos de "alucinação".

# Cache: {base_dir_resolvido: {numero_20_digitos: "nome_do_arquivo.csv"}}
_INDICE_BASE_CJF: dict[str, dict[str, str]] = {}

# Colunas de CSV que podem conter o número do processo (ordem de preferência).
_COLUNAS_NUMERO_PROCESSO = ("PROCESSO JUDICIAL", "Nº", "processo", "numero")


def limpar_indice_base() -> None:
    """Limpa o índice em memória dos números de processo da base jurídica."""
    _INDICE_BASE_CJF.clear()


def _indexar_numeros_base(base_dir: str, varredura_ampla: bool = False) -> dict[str, str]:
    """
    Varre os CSVs da base jurídica e mapeia cada número de processo (20 dígitos)
    para o nome do arquivo em que aparece. O resultado é cacheado por base_dir.

    Por padrão lê apenas as colunas conhecidas de número do processo
    (`_COLUNAS_NUMERO_PROCESSO`) — rápido e suficiente, pois é dessas colunas que
    o RAG extrai o número citado (vão para o cabeçalho do chunk). Isso evita
    varrer campos enormes (ex.: DECISÃO COMPLETA) em bases de centenas de MB.

    Se `varredura_ampla=True`, também procura números embutidos em qualquer
    campo da linha via regex CNJ (mais abrangente, porém bem mais lento).
    """
    import csv as _csv
    from pathlib import Path as _Path

    chave = str(_Path(base_dir).resolve()) if base_dir else ""
    if chave in _INDICE_BASE_CJF:
        return _INDICE_BASE_CJF[chave]

    indice: dict[str, str] = {}
    base = _Path(base_dir) if base_dir else None
    if base and base.exists():
        # Aumenta o limite de campo do csv: ementas/decisões são muito longas.
        try:
            _csv.field_size_limit(10 * 1024 * 1024)
        except (OverflowError, ValueError):
            pass

        for arquivo in sorted(base.glob("**/*.csv")):
            nome = arquivo.name
            try:
                with open(arquivo, "r", encoding="utf-8-sig", newline="") as f:
                    # Detecta o delimitador de forma simples (mesma lógica do pipeline).
                    amostra = f.read(4096)
                    f.seek(0)
                    delimitador = ";"
                    try:
                        dialect = _csv.Sniffer().sniff(amostra, delimiters=",;\t")
                        delimitador = dialect.delimiter
                    except _csv.Error:
                        primeira = amostra.splitlines()[0] if amostra else ""
                        delimitador = max(
                            [",", ";", "\t"], key=lambda d: primeira.count(d)
                        )
                    reader = _csv.DictReader(f, delimiter=delimitador)
                    for row in reader:
                        # 1) Colunas conhecidas (rápido e confiável).
                        for coluna in _COLUNAS_NUMERO_PROCESSO:
                            valor = row.get(coluna)
                            if not valor:
                                continue
                            dig = _normalizar_numero(str(valor))
                            if len(dig) == 20:
                                indice.setdefault(dig, nome)
                        # 2) Varredura ampla opcional (captura números embutidos
                        #    em qualquer campo; custosa em bases grandes).
                        if varredura_ampla:
                            linha_texto = " ".join(
                                str(v) for v in row.values() if v is not None
                            )
                            for regex in (_RE_CNJ_MASCARA, _RE_CNJ_DIGITOS):
                                for m in regex.findall(linha_texto):
                                    dig = _normalizar_numero(m)
                                    if len(dig) == 20:
                                        indice.setdefault(dig, nome)
            except (OSError, UnicodeDecodeError):
                # Um CSV ilegível não deve derrubar a validação inteira.
                continue

    _INDICE_BASE_CJF[chave] = indice
    return indice


# Códigos HTTP que justificam nova tentativa com espera (servidor saturado).
_HTTP_RETRIAVEIS = {429, 503}


def _consultar_datajud(
    alias: str,
    numero_digitos: str,
    timeout: float,
    max_retries_429: int = 3,
    base_delay_429_s: float = 2.0,
    max_delay_429_s: float = 15.0,
) -> dict[str, Any]:
    """
    Faz o POST _search no DataJud e retorna o JSON de resposta.

    A API do DataJud é pública, compartilhada e pode responder HTTP 429
    (Too Many Requests) ou 503 sob carga. Aplicamos um backoff LEVE nesses
    casos: espera crescente (respeitando o header `Retry-After`, quando
    presente) e repete algumas vezes antes de propagar o erro. Também há um
    retry para timeouts/erros de rede transitórios.
    """
    url = f"{_BASE_URL}/{alias}/_search"
    body = json.dumps({"query": {"match": {"numeroProcesso": numero_digitos}}}).encode(
        "utf-8"
    )

    def _nova_req() -> urllib.request.Request:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"APIKey {_api_key()}")
        req.add_header("Content-Type", "application/json")
        return req

    tentativas_429 = 0
    tentativas_rede = 0
    while True:
        try:
            with urllib.request.urlopen(_nova_req(), timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 429/503: backoff leve e tenta de novo (até max_retries_429).
            if exc.code in _HTTP_RETRIAVEIS and tentativas_429 < max_retries_429:
                espera = min(
                    base_delay_429_s * (2 ** tentativas_429), max_delay_429_s
                )
                # Respeita o Retry-After do servidor, se vier (em segundos).
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        espera = min(max(float(retry_after), espera), max_delay_429_s)
                    except (TypeError, ValueError):
                        pass
                tentativas_429 += 1
                time.sleep(espera)
                continue
            # Outros HTTPError (ou 429 após esgotar as tentativas) propagam.
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            # Erro de rede transitório: um único retry imediato.
            if tentativas_rede < 1:
                tentativas_rede += 1
                continue
            raise exc


def _verificar_base_cjf(resultado: dict[str, Any], base_dir: str | None) -> None:
    """
    Enriquecе o `resultado` com a informação de se o número consta na base
    jurídica (CSV CJF). Preenche:
      - na_base_cjf (bool): True se o número foi encontrado em algum CSV.
      - arquivo_cjf (str|None): nome do arquivo onde foi encontrado.

    Esta checagem é LOCAL (não faz rede) e serve de segunda fonte de verdade:
    se o DataJud não achou mas o número está na base, a citação NÃO é uma
    alucinação — veio dos arquivos indexados. Não altera `existe` (que reflete
    só o DataJud); a camada de exibição decide como rotular combinando os dois.
    """
    resultado.setdefault("na_base_cjf", False)
    resultado.setdefault("arquivo_cjf", None)
    if not base_dir:
        return
    digitos = _normalizar_numero(resultado.get("numero", ""))
    if len(digitos) != 20:
        return
    indice = _indexar_numeros_base(base_dir)
    arquivo = indice.get(digitos)
    if arquivo:
        resultado["na_base_cjf"] = True
        resultado["arquivo_cjf"] = arquivo


def consultar_processo(
    numero: str,
    timeout: float = 30.0,
    palavras_tema: tuple | list | None = None,
    base_dir: str | None = None,
    preferir_base: bool = True,
) -> dict[str, Any]:
    """
    Consulta um processo no DataJud e valida o assunto.

    Args:
        numero: número do processo (CNJ, com ou sem máscara).
        timeout: tempo limite da requisição.
        palavras_tema: palavras-chave do tema para checar compatibilidade do
            assunto. Se None, usa as palavras padrão do módulo.
        base_dir: diretório da base jurídica. Quando informado, verifica-se se
            o número consta nos CSVs da base (CJF). Preenche os campos
            `na_base_cjf` e `arquivo_cjf`.
        preferir_base: quando True (padrão) e o número JÁ consta na base CJF,
            NÃO consulta o DataJud — a base é fonte de verdade suficiente para o
            RAG, e isso evita chamadas desnecessárias à API pública (reduzindo
            os erros 429/rate limit). Só vale quando `base_dir` é informado e,
            portanto, só afeta o RAG (a Padrão não passa `base_dir`).

    Retorna um dict com:
      numero, tribunal, existe, assuntos, compativel_tema, erro (opcional),
      na_base_cjf, arquivo_cjf.
    """
    digitos = _normalizar_numero(numero)

    # Cache: se já consultamos este número, reaproveita o resultado.
    # A compatibilidade de tema é recalculada com as palavras atuais, pois
    # pode variar entre temas mesmo com o mesmo processo em cache.
    if digitos in _CACHE:
        cacheado = dict(_CACHE[digitos])
        cacheado["numero"] = numero  # preserva o formato original citado
        if cacheado.get("existe"):
            assuntos_cache = [{"nome": n} for n in cacheado.get("assuntos", [])]
            cacheado["compativel_tema"] = _assunto_compativel(
                assuntos_cache, palavras_tema
            )
        # A checagem da base é local e depende do base_dir atual; aplica-se
        # sobre o resultado cacheado (não é guardada no _CACHE da API).
        _verificar_base_cjf(cacheado, base_dir)
        return cacheado

    resultado: dict[str, Any] = {
        "numero": numero,
        "tribunal": None,
        "existe": False,
        "assuntos": [],
        "compativel_tema": False,
        "erro": None,
        "na_base_cjf": False,
        "arquivo_cjf": None,
    }

    # OTIMIZAÇÃO (opção "preferir base"): se o número já consta na base CJF,
    # dispensamos a consulta ao DataJud. Para o RAG, constar na base já confirma
    # que a citação é legítima; evitar a chamada reduz a pressão sobre a API
    # pública (menos 429) e acelera a validação. `existe` fica False (não foi o
    # DataJud que confirmou), mas `na_base_cjf=True` sinaliza a origem.
    if preferir_base and base_dir:
        _verificar_base_cjf(resultado, base_dir)
        if resultado.get("na_base_cjf"):
            alias_local = _resolver_endpoint(digitos)
            if alias_local:
                resultado["tribunal"] = alias_local.replace("api_publica_", "").upper()
            # Sinaliza que o DataJud foi deliberadamente dispensado (não é erro).
            resultado["datajud_dispensado"] = True
            return resultado

    alias = _resolver_endpoint(digitos)
    if alias is None:
        resultado["erro"] = "Fora do escopo (não é processo de TRF/Justiça Federal)."
        _verificar_base_cjf(resultado, base_dir)
        return resultado
    resultado["tribunal"] = alias.replace("api_publica_", "").upper()

    try:
        dados = _consultar_datajud(alias, digitos, timeout)
    except urllib.error.HTTPError as exc:
        resultado["erro"] = f"HTTP {exc.code} ao consultar o DataJud."
        _verificar_base_cjf(resultado, base_dir)
        return resultado
    except (urllib.error.URLError, TimeoutError) as exc:
        resultado["erro"] = f"Falha de conexão com o DataJud: {exc}"
        _verificar_base_cjf(resultado, base_dir)
        return resultado
    except Exception as exc:  # noqa: BLE001
        resultado["erro"] = f"Erro inesperado: {exc}"
        _verificar_base_cjf(resultado, base_dir)
        return resultado

    hits = dados.get("hits", {}).get("hits", [])
    if not hits:
        resultado["existe"] = False
        # Cacheia apenas a parte vinda do DataJud (sem a flag local da base).
        _CACHE[digitos] = {k: v for k, v in resultado.items()
                           if k not in ("na_base_cjf", "arquivo_cjf")}
        _verificar_base_cjf(resultado, base_dir)
        return resultado

    resultado["existe"] = True
    source = hits[0].get("_source", {})
    assuntos = source.get("assuntos", []) or []
    resultado["assuntos"] = [str(a.get("nome", "")) for a in assuntos]
    resultado["compativel_tema"] = _assunto_compativel(assuntos, palavras_tema)
    _CACHE[digitos] = {k: v for k, v in resultado.items()
                       if k not in ("na_base_cjf", "arquivo_cjf")}
    _verificar_base_cjf(resultado, base_dir)
    return resultado


def validar_processos_citados(
    texto: str,
    timeout: float = 30.0,
    palavras_tema: tuple | list | None = None,
    base_dir: str | None = None,
    preferir_base: bool = True,
) -> list[dict[str, Any]]:
    """
    Extrai e valida todos os processos citados em um texto.

    palavras_tema permite adaptar a checagem de compatibilidade ao tema
    configurado; se None, usa as palavras padrão do módulo.

    base_dir ativa o fallback na base jurídica (CJF): processos presentes nos
    CSVs da base são marcados com na_base_cjf=True — indicando que a citação foi
    extraída da base e não inventada. Use-o apenas para a abordagem RAG (a LLM
    Padrão não tem acesso à base).

    preferir_base (padrão True): quando um número já consta na base CJF, pula a
    consulta ao DataJud (economiza chamadas à API pública e evita 429). Só tem
    efeito quando base_dir é informado.
    """
    numeros = extrair_numeros_processo(texto)
    return [
        consultar_processo(
            n,
            timeout=timeout,
            palavras_tema=palavras_tema,
            base_dir=base_dir,
            preferir_base=preferir_base,
        )
        for n in numeros
    ]
