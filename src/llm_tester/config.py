"""
Leitura dos parâmetros e chaves de API do llm-tester.

A configuração vem de duas fontes:
  1. config.json (parâmetros padrão do projeto)
  2. Variáveis de ambiente / .env (chave GROQ_API_KEY)

Também pode ser sobrescrita em tempo de execução (ex.: pela UI Streamlit).
"""

from __future__ import annotations

import os
import json
import copy
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv é opcional
    pass


# Configuração padrão embutida — usada caso config.json não exista.
DEFAULT_CONFIG: dict[str, Any] = {
    "projeto": {
        "nome": "llm-tester",
        "descricao": "Benchmarking de LLMs (Padrão vs. RAG) - Pensão por Morte",
    },
    "tema": {
        "nome": "Pensão por Morte",
        "area": "Direito Previdenciário",
        "dominio_prompt": "Direito Previdenciário",
        "especialidade_juiz": "Direito Previdenciário (tema Pensão por Morte)",
        "palavras_chave_datajud": [
            "pensao", "pensão", "morte", "previdenci",
            "previdenciário", "beneficio",
        ],
    },
    "api": {"groq_api_key_env": "GROQ_API_KEY"},
    "dados": {
        "base_juridica": "data/base_juridica/",
        "arquivo_perguntas": "data/perguntas/perguntas.json",
        "arquivo_gabarito": "data/gabarito/gabarito.json",
        "diretorio_saida": "outputs/",
    },
    "modelos": {
        "llm_execucao": "openai/gpt-oss-20b",
        "llm_juiz": "openai/gpt-oss-120b",
        "modelos_disponiveis": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound",
            "groq/compound-mini",
            "qwen/qwen3.6-27b",
        ],
    },
    "parametros": {
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 4,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "pensao_por_morte",
        "persist_directory": "data/chroma_db",
        "validar_processos_datajud": True,
    },
    "rate_limit": {"max_retries": 6, "base_delay_s": 2.0, "max_delay_s": 60.0},
    "prompts": {
        "llm_padrao": (
            "Responda de forma precisa à seguinte questão de {dominio}: "
            "{pergunta}"
        ),
        "llm_rag": (
            "Você é um assistente jurídico especializado em {dominio}. "
            "Responda à pergunta utilizando ESTRITAMENTE as informações "
            "fornecidas no contexto abaixo.\n\nCONTEXTO:\n"
            "{contexto_recuperado}\n\nPERGUNTA:\n{pergunta}"
        ),
    },
}


class Config:
    """Carrega e disponibiliza os parâmetros do llm-tester."""

    def __init__(self, config_path: str | None = "config.json"):
        self.config_path = Path(config_path) if config_path else None
        self.data: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

        if self.config_path and self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._merge(self.data, json.load(f))

    @staticmethod
    def _merge(base: dict, override: dict) -> None:
        """Merge recursivo de dicionários (override tem precedência)."""
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                Config._merge(base[k], v)
            else:
                base[k] = v

    # --- Acessores de conveniência -----------------------------------------

    @property
    def groq_api_key(self) -> str | None:
        """Retorna a chave da Groq API (env var ou definida em runtime)."""
        env_name = self.data["api"].get("groq_api_key_env", "GROQ_API_KEY")
        return self.data["api"].get("groq_api_key") or os.getenv(env_name)

    def set_groq_api_key(self, key: str) -> None:
        """Define a chave da Groq em runtime (usado pela UI)."""
        self.data["api"]["groq_api_key"] = key
        os.environ[self.data["api"].get("groq_api_key_env", "GROQ_API_KEY")] = key

    @property
    def tema(self) -> dict[str, Any]:
        return self.data.get("tema", {})

    @property
    def dados(self) -> dict[str, Any]:
        return self.data["dados"]

    @property
    def modelos(self) -> dict[str, Any]:
        return self.data["modelos"]

    @property
    def parametros(self) -> dict[str, Any]:
        return self.data["parametros"]

    @property
    def rate_limit(self) -> dict[str, Any]:
        return self.data["rate_limit"]

    @property
    def prompts(self) -> dict[str, Any]:
        return self.data["prompts"]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def salvar(self, caminho: str | Path | None = None) -> str:
        """
        Persiste a configuração atual em disco (JSON).

        Por segurança, a chave de API definida em runtime NÃO é gravada — ela
        deve permanecer no .env / variável de ambiente.
        """
        destino = Path(caminho) if caminho else (self.config_path or Path("config.json"))
        dados = copy.deepcopy(self.data)
        # Remove a chave de API para não vazar segredo no arquivo.
        if isinstance(dados.get("api"), dict):
            dados["api"].pop("groq_api_key", None)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        return str(destino)
