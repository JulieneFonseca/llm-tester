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


def _consultar_datajud(alias: str, numero_digitos: str, timeout: float) -> dict[str, Any]:
    """
    Faz o POST _search no DataJud e retorna o JSON de resposta.

    A API do DataJud pode ser lenta/instável; por isso há uma tentativa extra
    em caso de timeout de leitura antes de propagar o erro.
    """
    url = f"{_BASE_URL}/{alias}/_search"
    body = json.dumps({"query": {"match": {"numeroProcesso": numero_digitos}}}).encode(
        "utf-8"
    )

    ultima_exc: Exception | None = None
    for _ in range(2):  # 1 tentativa + 1 retry
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"APIKey {_api_key()}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            ultima_exc = exc
            continue
    raise ultima_exc  # type: ignore[misc]


def consultar_processo(
    numero: str,
    timeout: float = 30.0,
    palavras_tema: tuple | list | None = None,
) -> dict[str, Any]:
    """
    Consulta um processo no DataJud e valida o assunto.

    Args:
        numero: número do processo (CNJ, com ou sem máscara).
        timeout: tempo limite da requisição.
        palavras_tema: palavras-chave do tema para checar compatibilidade do
            assunto. Se None, usa as palavras padrão do módulo.

    Retorna um dict com:
      numero, tribunal, existe, assuntos, compativel_tema, erro (opcional).
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
        return cacheado

    resultado: dict[str, Any] = {
        "numero": numero,
        "tribunal": None,
        "existe": False,
        "assuntos": [],
        "compativel_tema": False,
        "erro": None,
    }

    alias = _resolver_endpoint(digitos)
    if alias is None:
        resultado["erro"] = "Fora do escopo (não é processo de TRF/Justiça Federal)."
        return resultado
    resultado["tribunal"] = alias.replace("api_publica_", "").upper()

    try:
        dados = _consultar_datajud(alias, digitos, timeout)
    except urllib.error.HTTPError as exc:
        resultado["erro"] = f"HTTP {exc.code} ao consultar o DataJud."
        return resultado
    except (urllib.error.URLError, TimeoutError) as exc:
        resultado["erro"] = f"Falha de conexão com o DataJud: {exc}"
        return resultado
    except Exception as exc:  # noqa: BLE001
        resultado["erro"] = f"Erro inesperado: {exc}"
        return resultado

    hits = dados.get("hits", {}).get("hits", [])
    if not hits:
        resultado["existe"] = False
        _CACHE[digitos] = dict(resultado)  # cacheia "não encontrado"
        return resultado

    resultado["existe"] = True
    source = hits[0].get("_source", {})
    assuntos = source.get("assuntos", []) or []
    resultado["assuntos"] = [str(a.get("nome", "")) for a in assuntos]
    resultado["compativel_tema"] = _assunto_compativel(assuntos, palavras_tema)
    _CACHE[digitos] = dict(resultado)  # cacheia resultado bem-sucedido
    return resultado


def validar_processos_citados(
    texto: str,
    timeout: float = 30.0,
    palavras_tema: tuple | list | None = None,
) -> list[dict[str, Any]]:
    """
    Extrai e valida todos os processos citados em um texto.

    palavras_tema permite adaptar a checagem de compatibilidade ao tema
    configurado; se None, usa as palavras padrão do módulo.
    """
    numeros = extrair_numeros_processo(texto)
    return [
        consultar_processo(n, timeout=timeout, palavras_tema=palavras_tema)
        for n in numeros
    ]
