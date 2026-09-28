"""
Reindexação da base jurídica no ChromaDB (sem rodar o benchmark/LLMs).

Modos de uso:

  # Reindexa TODA a base do zero (apaga e reprocessa tudo):
  python reindexar.py

  # Indexa/atualiza arquivos específicos (incremental — mantém o resto):
  python reindexar.py --arquivos "a.pdf" "b.csv"
  python reindexar.py --arquivo "nome do arquivo.pdf"      # atalho p/ um só

  # Zera a base e reindexa APENAS os arquivos selecionados:
  python reindexar.py --arquivos "a.pdf" "b.csv" --zerar

A indexação ocorre em lotes (index_batch_size no config.json) para não
estourar a memória. O progresso é impresso no stdout e gravado em
_reindex_log.txt para acompanhamento confiável.
"""
from __future__ import annotations

import os
import sys
import time
import argparse

# Desativa a telemetria do ChromaDB ANTES de qualquer import do chromadb.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# Garante saída UTF-8 mesmo em terminais Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

from llm_tester.config import Config  # noqa: E402
from llm_tester.pipeline import Pipeline  # noqa: E402


def _resolver_arquivo(nome: str, base_dir: str) -> str:
    """
    Aceita tanto um caminho completo quanto só o nome do arquivo. Se for só o
    nome, procura dentro da base jurídica (recursivamente).
    """
    from pathlib import Path

    p = Path(nome)
    if p.exists():
        return str(p)

    base = Path(base_dir)
    candidatos = list(base.glob(f"**/{nome}"))
    if candidatos:
        return str(candidatos[0])

    raise FileNotFoundError(
        f"Arquivo '{nome}' não encontrado nem como caminho nem dentro de {base_dir}."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Reindexação do ChromaDB")
    ap.add_argument(
        "--arquivo",
        default=None,
        help="Atalho para indexar um único arquivo (nome ou caminho).",
    )
    ap.add_argument(
        "--arquivos",
        nargs="+",
        default=None,
        help="Lista de arquivos a indexar (nomes ou caminhos, separados por espaço).",
    )
    ap.add_argument(
        "--zerar",
        action="store_true",
        help="Zera TODA a base antes de reindexar (usado com --arquivos/--arquivo). "
             "Sem --arquivos, reindexa a base completa do zero.",
    )
    args = ap.parse_args()

    # Unifica --arquivo e --arquivos numa lista só.
    selecionados = list(args.arquivos) if args.arquivos else []
    if args.arquivo:
        selecionados.append(args.arquivo)

    log_file = open("_reindex_log.txt", "w", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log("[init] Carregando configuração...")
        config = Config("config.json")
        base_dir = config.dados["base_juridica"]
        log(f"[init] Base jurídica: {base_dir}")
        log(f"[init] Batch size: {config.parametros.get('index_batch_size')}")

        pipeline = Pipeline(config)
        t0 = time.perf_counter()

        if selecionados:
            caminhos = [_resolver_arquivo(n, base_dir) for n in selecionados]
            modo = "ZERAR + selecionados" if args.zerar else "incremental (selecionados)"
            log(f"[init] Modo: {modo} — {len(caminhos)} arquivo(s).")
            n = pipeline.indexar_selecionados(caminhos, log=log, zerar=args.zerar)
            escopo = f"{len(caminhos)} arquivo(s) selecionado(s)"
        else:
            log("[init] Modo COMPLETO — reindexando toda a base do zero.")
            n = pipeline.indexar_base(base_dir, log=log, forcar=True)
            escopo = "base completa"

        dt = time.perf_counter() - t0
        log(f"\n[FIM] Indexação concluída ({escopo}): {n} chunks em {dt:.0f}s.")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        log(f"\n[ERRO] {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return 1
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
