"""
Ponto de entrada CLI (modo headless) do llm-tester.

Uso:
  python main.py --config config.json
  python main.py --config config.json --reindexar
  python main.py --modelo llama-3.1-8b-instant --juiz llama-3.3-70b-versatile
"""

from __future__ import annotations

import sys
import argparse

# Permite executar tanto como script quanto como módulo
sys.path.insert(0, "src")

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from llm_tester.config import Config  # noqa: E402
from llm_tester.pipeline import executar_benchmarking  # noqa: E402
from llm_tester import utils  # noqa: E402

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="llm-tester — Benchmarking de LLMs (Padrão vs. RAG)"
    )
    parser.add_argument("--config", default="config.json", help="Arquivo de configuração")
    parser.add_argument("--modelo", default=None, help="LLM de execução (override)")
    parser.add_argument("--juiz", default=None, help="LLM juiz (override)")
    parser.add_argument("--reindexar", action="store_true", help="Força reindexação da base")
    args = parser.parse_args()

    config = Config(args.config)

    tema = config.tema
    nome_tema = tema.get("nome", "Pensão por Morte")
    area_tema = tema.get("area", "Direito Previdenciário")

    console.print(Panel(
        f"[bold blue]llm-tester[/bold blue]\n"
        f"Benchmarking de LLMs (Padrão vs. RAG) - {area_tema} / {nome_tema}\n"
        f"Modelo de execução: {args.modelo or config.modelos['llm_execucao']}\n"
        f"Modelo juiz: {args.juiz or config.modelos['llm_juiz']}",
        title="llm-tester", border_style="blue",
    ))

    if not config.groq_api_key:
        console.print(
            "\n[bold red]GROQ_API_KEY não configurada.[/bold red]\n"
            "   Defina a variável de ambiente GROQ_API_KEY ou crie um arquivo .env.",
        )
        return 1

    perguntas = utils.carregar_perguntas(config.dados["arquivo_perguntas"])
    gabarito = utils.carregar_gabarito(config.dados["arquivo_gabarito"])
    console.print(f"\n{len(perguntas)} perguntas | {len(gabarito)} itens de gabarito\n")

    relatorio = executar_benchmarking(
        config=config,
        perguntas=perguntas,
        gabarito=gabarito,
        modelo_execucao=args.modelo,
        modelo_juiz=args.juiz,
        log=lambda m: console.print(m),
        reindexar=args.reindexar,
    )

    caminho_json = utils.salvar_json(relatorio, config.dados["diretorio_saida"])
    caminho_csv = utils.salvar_csv(relatorio, config.dados["diretorio_saida"])

    # Resumo de métricas
    m = relatorio["execucao_metadata"]["metricas_qualidade_global"]
    tabela = Table(title="Métricas Globais", show_lines=True)
    tabela.add_column("Métrica")
    tabela.add_column("Padrão", justify="right")
    tabela.add_column("RAG", justify="right")

    def _t(v: float) -> str:
        return f"{v:.2f} s"

    def _n(v: float) -> str:
        return f"{v:.2f}"

    def _p(v: float) -> str:
        return f"{v:.1f}%"

    tabela.add_row("Tempo médio", _t(m["tempo_medio_llm_padrao_s"]), _t(m["tempo_medio_llm_rag_s"]))
    tabela.add_row("Fidelidade média", _n(m["nota_fidelidade_media_padrao"]), _n(m["nota_fidelidade_media_rag"]))
    tabela.add_row("Precisão média", _n(m["nota_precisao_media_padrao"]), _n(m["nota_precisao_media_rag"]))
    tabela.add_row("Completude média", _n(m["nota_completude_media_padrao"]), _n(m["nota_completude_media_rag"]))
    tabela.add_row("Alucinação", _p(m["taxa_alucinacao_padrao_pct"]), _p(m["taxa_alucinacao_rag_pct"]))
    console.print(tabela)
    console.print(
        f"[dim]Overhead médio do RAG: {m['overhead_medio_rag_s']:.2f} s "
        f"({m['overhead_medio_rag_pct']:.1f}%)[/dim]"
    )

    console.print(Panel(
        f"[bold green]Benchmarking concluído![/bold green]\n"
        f"JSON: {caminho_json}\n"
        f"CSV:  {caminho_csv}",
        title="Finalizado", border_style="green",
    ))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\nInterrompido pelo usuário.", style="yellow")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"\nErro fatal: {exc}", style="bold red")
        sys.exit(1)
