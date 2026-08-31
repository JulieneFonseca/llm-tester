"""
Módulo de avaliação automatizada (LLM Juiz).

A LLM Juiz recebe a pergunta, o gabarito oficial e as duas respostas
(Padrão e RAG) e retorna, em JSON estrito, notas de qualidade jurídica,
detecção de alucinação e justificativas.

Usa o parâmetro response_format={"type": "json_object"} da Groq API para
garantir um JSON parseável.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import Config
from .utils import com_backoff


SYSTEM_PROMPT_JUIZ_TEMPLATE = """Você é um Juiz avaliador especializado em {especialidade}.

Sua tarefa é avaliar DUAS respostas geradas por sistemas de IA para a MESMA pergunta,
comparando-as com um GABARITO OFICIAL de referência.

Avalie CADA resposta segundo os critérios (escala 0 a 5):
- nota_fidelidade: fidelidade às fontes jurídicas / ausência de invenção de conteúdo.
- nota_precisao_gabarito: quão próxima a resposta está do gabarito oficial.
- nota_completude: cobertura dos pontos essenciais do gabarito.
- alucinacao_detectada (booleano): true se a resposta contém informação incorreta
  ou não suportada (jurisprudência inexistente, dispositivo legal errado, etc.).
- fundamentacao_correta (booleano): true se cita/fundamenta corretamente
  (súmula, artigo de lei, acórdão) de forma condizente com o gabarito.
- justificativa_juiz: explicação objetiva e curta da avaliação. Ao se referir
  às respostas na justificativa, use SEMPRE os termos "a resposta da LLM Padrão"
  e "a resposta da LLM com RAG". NUNCA use "resposta A", "resposta B", "A" ou "B".

Escala de notas:
0 = totalmente incorreto | 1 = majoritariamente incorreto | 2 = parcialmente correto |
3 = correto nos pontos principais | 4 = correto e completo | 5 = totalmente fiel e preciso.

Responda EXCLUSIVAMENTE com um objeto JSON válido no formato:
{
  "avaliacao_llm_padrao": {
    "nota_fidelidade": <0-5>,
    "nota_precisao_gabarito": <0-5>,
    "nota_completude": <0-5>,
    "alucinacao_detectada": <true|false>,
    "fundamentacao_correta": <true|false>,
    "justificativa_juiz": "<texto>"
  },
  "avaliacao_llm_rag": {
    "nota_fidelidade": <0-5>,
    "nota_precisao_gabarito": <0-5>,
    "nota_completude": <0-5>,
    "alucinacao_detectada": <true|false>,
    "fundamentacao_correta": <true|false>,
    "justificativa_juiz": "<texto>"
  }
}
"""


def _avaliacao_vazia(justificativa: str) -> dict[str, Any]:
    return {
        "nota_fidelidade": 0,
        "nota_precisao_gabarito": 0,
        "nota_completude": 0,
        "alucinacao_detectada": True,
        "fundamentacao_correta": False,
        "justificativa_juiz": justificativa,
    }


class Judge:
    """LLM Juiz — avalia as respostas Padrão e RAG e mede o tempo de avaliação."""

    def __init__(self, config: Config, modelo_juiz: str | None = None):
        self.config = config
        self.rate = config.rate_limit
        self.params = config.parametros
        self.modelo = modelo_juiz or config.modelos["llm_juiz"]
        self._llm = None
        self._llm_texto = None

        # System prompt do Juiz montado a partir do tema configurado.
        # Usa replace (e não .format) porque o template contém chaves {} de JSON.
        especialidade = config.tema.get(
            "especialidade_juiz", "Direito Previdenciário (tema Pensão por Morte)"
        )
        self.system_prompt = SYSTEM_PROMPT_JUIZ_TEMPLATE.replace(
            "{especialidade}", especialidade
        )

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self.modelo,
                temperature=0.0,
                api_key=self.config.groq_api_key,
                max_retries=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        return self._llm

    def avaliar(
        self,
        pergunta: str,
        gabarito: str,
        resposta_padrao: str,
        resposta_rag: str,
        on_wait: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Avalia ambas as respostas. Retorna:
          {
            "avaliacao_llm_padrao": {...},
            "avaliacao_llm_rag": {...},
            "tempo_avaliacao_juiz_s": <float>
          }
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        user_msg = f"""## PERGUNTA:
{pergunta}

## GABARITO OFICIAL:
{gabarito}

## RESPOSTA DA LLM PADRÃO (sem contexto):
{resposta_padrao}

## RESPOSTA DA LLM COM RAG (com contexto jurídico):
{resposta_rag}

Avalie a resposta da LLM Padrão em "avaliacao_llm_padrao" e a resposta da LLM com RAG em "avaliacao_llm_rag".
"""
        mensagens = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ]

        llm = self._get_llm()

        def chamada():
            inicio = time.perf_counter()
            resp = llm.invoke(mensagens)
            decorrido = time.perf_counter() - inicio
            return resp.content, decorrido

        content, tempo = com_backoff(
            chamada,
            max_retries=self.rate.get("max_retries", 6),
            base_delay_s=self.rate.get("base_delay_s", 2.0),
            max_delay_s=self.rate.get("max_delay_s", 60.0),
            on_wait=on_wait,
        )

        avaliacao = self._parse_json(content)
        avaliacao["tempo_avaliacao_juiz_s"] = round(tempo, 4)
        return avaliacao

    def _get_llm_texto(self):
        """LLM do Juiz para saída em TEXTO livre (sem response_format JSON)."""
        if self._llm_texto is None:
            from langchain_groq import ChatGroq

            self._llm_texto = ChatGroq(
                model=self.modelo,
                temperature=0.2,
                api_key=self.config.groq_api_key,
                max_retries=0,
            )
        return self._llm_texto

    def parecer_final(
        self,
        metricas: dict[str, Any],
        resultados: list[dict],
        on_wait: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Gera um parecer descritivo final da LLM Juiz, consolidando todas as
        avaliações e posicionando qual abordagem (LLM Padrão vs LLM com RAG)
        teve melhor desempenho geral.

        Retorna:
          {
            "abordagem_vencedora": "RAG" | "Padrão" | "Empate",
            "parecer_texto": "<texto descritivo>",
            "tempo_parecer_juiz_s": <float>
          }
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        # Resumo compacto por pergunta (para o Juiz raciocinar sobre os casos).
        linhas_casos = []
        for r in resultados:
            av_p = r.get("avaliacao_llm_padrao", {})
            av_r = r.get("avaliacao_llm_rag", {})
            linhas_casos.append(
                f"- Pergunta {r.get('id_pergunta')}: "
                f"Padrão [fid={av_p.get('nota_fidelidade')}, "
                f"prec={av_p.get('nota_precisao_gabarito')}, "
                f"compl={av_p.get('nota_completude')}, "
                f"alucinação={'sim' if av_p.get('alucinacao_detectada') else 'não'}] | "
                f"RAG [fid={av_r.get('nota_fidelidade')}, "
                f"prec={av_r.get('nota_precisao_gabarito')}, "
                f"compl={av_r.get('nota_completude')}, "
                f"alucinação={'sim' if av_r.get('alucinacao_detectada') else 'não'}]"
            )
        resumo_casos = "\n".join(linhas_casos)

        resumo_metricas = f"""Médias globais (escala 0 a 5):
- Fidelidade: Padrão={metricas.get('nota_fidelidade_media_padrao')} | RAG={metricas.get('nota_fidelidade_media_rag')}
- Precisão vs. gabarito: Padrão={metricas.get('nota_precisao_media_padrao')} | RAG={metricas.get('nota_precisao_media_rag')}
- Completude: Padrão={metricas.get('nota_completude_media_padrao')} | RAG={metricas.get('nota_completude_media_rag')}
- Taxa de alucinação: Padrão={metricas.get('taxa_alucinacao_padrao_pct')}% | RAG={metricas.get('taxa_alucinacao_rag_pct')}%
- Tempo médio: Padrão={metricas.get('tempo_medio_llm_padrao_s')}s | RAG={metricas.get('tempo_medio_llm_rag_s')}s
- Overhead do RAG: {metricas.get('overhead_medio_rag_s')}s ({metricas.get('overhead_medio_rag_pct')}%)"""

        system = (
            f"Você é um Juiz avaliador especializado em "
            f"{self.config.tema.get('especialidade_juiz', 'Direito Previdenciário')}. "
            "Escreva um parecer final consolidado, técnico e objetivo, em português, "
            "comparando duas abordagens de IA jurídica: a LLM Padrão (sem contexto) e "
            "a LLM com RAG (com contexto jurídico recuperado). "
            "Baseie-se estritamente nas métricas e nos casos fornecidos."
        )

        user = f"""Com base nas avaliações do benchmarking abaixo, escreva um PARECER FINAL.

{resumo_metricas}

Detalhe por pergunta:
{resumo_casos}

O parecer deve conter, em texto corrido (3 a 5 parágrafos):
1. Qual abordagem teve MELHOR desempenho geral e por quê (cite as métricas).
2. Pontos fortes e fracos de cada abordagem observados nos casos.
3. A questão central: o ganho de qualidade do RAG justifica o overhead de latência?
4. Uma recomendação prática de qual abordagem usar neste domínio.

Comece a resposta com uma linha exatamente no formato:
VENCEDORA: <RAG|Padrão|Empate>

Em seguida, o texto do parecer."""

        llm = self._get_llm_texto()

        def chamada():
            inicio = time.perf_counter()
            resp = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            decorrido = time.perf_counter() - inicio
            return resp.content, decorrido

        content, tempo = com_backoff(
            chamada,
            max_retries=self.rate.get("max_retries", 6),
            base_delay_s=self.rate.get("base_delay_s", 2.0),
            max_delay_s=self.rate.get("max_delay_s", 60.0),
            on_wait=on_wait,
        )

        # Extrai a abordagem vencedora da primeira linha "VENCEDORA: ..."
        abordagem = "Indefinido"
        texto = (content or "").strip()
        for linha in texto.splitlines():
            if linha.strip().upper().startswith("VENCEDORA:"):
                valor = linha.split(":", 1)[1].strip()
                abordagem = valor or "Indefinido"
                break

        return {
            "abordagem_vencedora": abordagem,
            "parecer_texto": texto,
            "tempo_parecer_juiz_s": round(tempo, 4),
        }

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parseia o JSON do juiz, com fallback tolerante."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}") + 1
            try:
                data = json.loads(content[start:end]) if start != -1 else {}
            except json.JSONDecodeError:
                data = {}

        if "avaliacao_llm_padrao" not in data:
            data["avaliacao_llm_padrao"] = _avaliacao_vazia(
                "Falha ao parsear avaliação do juiz."
            )
        if "avaliacao_llm_rag" not in data:
            data["avaliacao_llm_rag"] = _avaliacao_vazia(
                "Falha ao parsear avaliação do juiz."
            )
        return data
