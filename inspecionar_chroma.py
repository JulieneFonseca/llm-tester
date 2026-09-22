"""
Inspeciona o conteúdo do ChromaDB (data/chroma_db).

Uso:
  python inspecionar_chroma.py                 # resumo + amostras
  python inspecionar_chroma.py --n 10          # 10 amostras
  python inspecionar_chroma.py --busca "pensao por morte"   # busca por texto

Observação: rode SOMENTE quando a indexação tiver terminado (evita lock do SQLite).
"""
from __future__ import annotations

import argparse
import json
import sys

# Lê os parâmetros do config.json para achar o diretório/collection corretos.
def carregar_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    params = cfg.get("parametros", {})
    persist_dir = params.get("persist_directory", "data/chroma_db")
    collection = params.get("collection_name", "pensao_por_morte")
    return persist_dir, collection


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspeciona o ChromaDB")
    ap.add_argument("--n", type=int, default=5, help="Número de amostras a exibir")
    ap.add_argument("--busca", default=None, help="Faz uma busca semântica por este texto")
    ap.add_argument("--chars", type=int, default=300, help="Máx. de chars por trecho exibido")
    args = ap.parse_args()

    persist_dir, collection_name = carregar_config()

    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)

    print("=" * 70)
    print(f"Diretório : {persist_dir}")
    print("Collections encontradas:")
    for c in client.list_collections():
        print(f"  - {c.name}")
    print("=" * 70)

    try:
        col = client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[erro] Não achei a collection '{collection_name}': {exc}")
        return 1

    total = col.count()
    print(f"Collection : {collection_name}")
    print(f"Total de chunks indexados: {total}")
    print("=" * 70)

    if total == 0:
        print("A collection está vazia (indexação ainda não concluída?).")
        return 0

    if args.busca:
        print(f"\n[BUSCA] '{args.busca}'  (top {args.n})\n")
        res = col.query(query_texts=[args.busca], n_results=args.n)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            origem = (meta or {}).get("source", "?")
            print(f"--- Resultado {i} | distância={dist:.4f} | fonte={origem}")
            print((doc or "")[: args.chars].replace("\n", " "))
            print()
    else:
        print(f"\n[AMOSTRAS] Primeiros {args.n} chunks:\n")
        res = col.get(limit=args.n, include=["documents", "metadatas"])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            origem = (meta or {}).get("source", "?")
            print(f"--- Chunk {i} | fonte={origem}")
            print((doc or "")[: args.chars].replace("\n", " "))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
