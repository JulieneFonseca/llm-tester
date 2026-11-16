"""
Importa o gabarito a partir de um documento Word (.docx) e gera
o gabarito.json.

Formato esperado (parágrafos, sem tabelas):
    Questão NN – título
    Pergunta: ...
    Resposta esperada: ... (inclui "O sentimento é X")
    Ementa esperada: ...
    Julgados de referência: ... (ignorado)
    Referência: ...
    Pontos obrigatórios: ... (opcional)
    Inaceitável: ...
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

RE_Q = re.compile(r"^Quest[aã]o\s+(\d+)\s*[–—-]\s*(.*)$", re.I)
_LABELS = {
    "pergunta": re.compile(r"^Pergunta\s*:\s*", re.I),
    "resposta": re.compile(r"^Resposta esperada\s*:\s*", re.I),
    "ementa": re.compile(r"^Ementa esperada\s*:\s*", re.I),
    "julgados": re.compile(r"^Julgados de refer[eê]ncia\s*:\s*", re.I),
    "referencias": re.compile(r"^Refer[eê]ncias?\s*:\s*", re.I),
    "pontos": re.compile(r"^Pontos obrigat[oó]rios\s*:\s*", re.I),
    "inaceitavel": re.compile(r"^Inaceit[aá]ve(?:l|is)\s*:\s*", re.I),
}
_RE_ROTULO_DESCONHECIDO = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ ]{2,40}:\s*$")
_RE_SENT = re.compile(
    r"sentimento\b[^.;\n]{0,200}?\b(positivo|negativo|inconclusivo|neutro)\b", re.I
)
_RE_JULG = re.compile(r"(\d+)\s+decis[õo]es", re.I)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()


def ler_paragrafos(conteudo: bytes) -> list[str]:
    """Extrai o texto dos parágrafos de um .docx."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        raiz = ET.fromstring(z.read("word/document.xml"))
    paragrafos = []
    for p in raiz.iter(f"{W}p"):
        partes = []
        for el in p.iter():
            if el.tag == f"{W}t":
                partes.append(el.text or "")
            elif el.tag in (f"{W}tab", f"{W}br"):
                partes.append(" ")
        t = _norm("".join(partes))
        if t:
            paragrafos.append(t)
    return paragrafos


def _itens(texto: str) -> list[str]:
    partes = re.split(r"\n+", texto)
    saida = []
    for p in partes:
        p = _norm(p).rstrip(";").strip()
        if p:
            saida.append(p)
    return saida


def parsear(conteudo: bytes) -> tuple[list[dict], list[str]]:
    """Converte o .docx em itens de gabarito. Retorna (itens, avisos)."""
    paragrafos = ler_paragrafos(conteudo)
    blocos: list[dict] = []
    atual = None
    campo = None
    for par in paragrafos:
        m = RE_Q.match(par)
        if m:
            atual = {"num": int(m.group(1)), "titulo": m.group(2), "campos": {}}
            blocos.append(atual)
            campo = None
            continue
        if atual is None:
            continue
        achou = False
        for nome, rx in _LABELS.items():
            mm = rx.match(par)
            if mm:
                campo = nome
                atual["campos"].setdefault(campo, [])
                resto = par[mm.end():].strip()
                if resto:
                    atual["campos"][campo].append(resto)
                achou = True
                break
        if achou:
            continue
        if _RE_ROTULO_DESCONHECIDO.match(par):
            campo = None  # rótulo desconhecido: ignora seu conteúdo
            continue
        if campo:
            atual["campos"][campo].append(par)

    itens: list[dict] = []
    avisos: list[str] = []
    for b in blocos:
        c = b["campos"]
        qid = f"Q{b['num']:02d}"
        pergunta = _norm(" ".join(c.get("pergunta", [])))
        resposta = "\n".join(c.get("resposta", [])).strip()
        ementa = "\n".join(c.get("ementa", [])).strip()
        refs = [i for t in c.get("referencias", []) for i in _itens(t)]
        inac = [i for t in c.get("inaceitavel", []) for i in _itens(t)]
        pontos = [i for t in c.get("pontos", []) for i in _itens(t)]

        if not pergunta:
            avisos.append(f"{qid}: sem 'Pergunta'.")
        if not resposta:
            avisos.append(f"{qid}: sem 'Resposta esperada'.")
        if not refs:
            avisos.append(f"{qid}: sem 'Referência'.")

        sent = None
        ms = _RE_SENT.search(resposta)
        if ms:
            sent = ms.group(1).lower()
        else:
            avisos.append(f"{qid}: sentimento não encontrado na resposta.")

        mj = _RE_JULG.search(pergunta) or _RE_JULG.search(" ".join(inac))
        julg = int(mj.group(1)) if mj else 3
        if not mj:
            avisos.append(f"{qid}: nº de julgados não identificado (usando 3).")

        item = {
            "id": qid,
            "pergunta": pergunta,
            "resposta_esperada": resposta,
        }
        if ementa:
            item["ementa_esperada"] = ementa
        item["referencias"] = refs
        item["criterios_inaceitaveis"] = inac
        if sent:
            item["sentimento_esperado"] = sent
        item["julgados_solicitados"] = julg
        if pontos:
            item["pontos_obrigatorios"] = pontos
        item["gabarito_oficial"] = resposta
        itens.append(item)

    if not itens:
        avisos.append("Nenhuma 'Questão NN – ...' encontrada no documento.")
    return itens, avisos


def fazer_backup(caminho: str | Path) -> str | None:
    """Copia o arquivo existente para <arquivo>.bak. Retorna o caminho do .bak."""
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    bak = caminho.with_name(caminho.name + ".bak")
    shutil.copy2(caminho, bak)
    return str(bak)


def gravar_arquivos(itens: list[dict], caminho_json: str | Path) -> dict:
    """
    Faz backup (.bak) e grava o gabarito.json.
    Retorna {"json": ..., "backups": [...]}.
    """
    from . import utils

    caminho_json = Path(caminho_json)
    backups = [b for b in (fazer_backup(caminho_json),) if b]
    utils.salvar_gabarito(itens, caminho_json)
    return {"json": str(caminho_json), "backups": backups}