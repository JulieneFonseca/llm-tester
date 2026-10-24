"""
Módulo de avaliação automatizada (LLM Juiz).

A LLM Juiz recebe a pergunta, o gabarito oficial e as duas respostas
(Padrão e RAG) e retorna, em JSON estrito, notas de qualidade jurídica,
sentimento processual, erros críticos, aferição de julgados e justificativas.

Usa o parâmetro response_format={"type": "json_object"} da Groq API para
garantir um JSON parseável.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import Config
from .utils import com_backoff


# Aproximação de tokens: para PT-BR, ~4 caracteres por token é uma estimativa
# conservadora e barata (sem depender de tokenizer externo).
_CHARS_POR_TOKEN = 4


def _estimar_tokens(texto: str) -> int:
    """Estima o número de tokens de um texto (aprox. 4 chars/token)."""
    if not texto:
        return 0
    return (len(texto) + _CHARS_POR_TOKEN - 1) // _CHARS_POR_TOKEN


def _truncar_por_tokens(texto: str, max_tokens: int) -> tuple[str, bool]:
    """
    Trunca `texto` para caber em `max_tokens` (aprox.), preservando o INÍCIO
    (que costuma conter o essencial da resposta jurídica). Retorna
    (texto_possivelmente_truncado, foi_truncado).
    """
    if max_tokens <= 0:
        return "", bool(texto)
    limite_chars = max_tokens * _CHARS_POR_TOKEN
    if len(texto) <= limite_chars:
        return texto, False
    corte = texto[:limite_chars].rstrip()
    return corte + "\n\n[...trecho truncado para respeitar o limite de tokens...]", True


def _e_request_too_large(exc: Exception) -> bool:
    """
    Detecta o erro 413 da Groq ('request too large' / limite de TPM excedido),
    independentemente do tipo exato da exceção lançada pelo cliente.
    """
    txt = str(exc).lower()
    return (
        "413" in txt
        or "request too large" in txt
        or "reduce your message size" in txt
        or ("rate_limit" in txt and "tokens" in txt and "tpm" in txt)
    )


def _e_json_validate_failed(exc: Exception) -> bool:
    """
    Detecta o erro 400 da Groq ('json_validate_failed'): o modo json_object
    não conseguiu validar a geração como JSON (tipicamente geração vazia porque
    o modelo gpt-oss gastou o orçamento de tokens no raciocínio interno).
    """
    txt = str(exc).lower()
    return (
        "json_validate_failed" in txt
        or "failed to validate json" in txt
    )


SYSTEM_PROMPT_JUIZ_TEMPLATE = """# PAPEL

Você é um avaliador especializado em {especialidade}. Sua função é avaliar
comparativamente as respostas produzidas por uma LLM na abordagem Padrão e na
abordagem com RAG, utilizando EXCLUSIVAMENTE os critérios definidos neste
instrumento.

Para cada questão você recebe:
- PERGUNTA: enunciado submetido à LLM.
- RESPOSTA ESPERADA: gabarito jurídico de referência.
- REFERÊNCIA: legislação, súmulas, julgados e fontes jurídicas indicadas.
- INACEITÁVEL: erros ou conclusões expressamente vedados.
- RESPOSTA DA LLM PADRÃO: resposta gerada sem contexto jurídico recuperado.
- RESPOSTA DA LLM COM RAG: resposta gerada com contexto recuperado da base.

# PRINCÍPIOS DE AVALIAÇÃO

1. Avalie cada resposta individualmente, com a mesma escala e os mesmos critérios.
2. Não considere uma resposta correta apenas por usar palavras semelhantes às do gabarito.
3. Considere a conclusão jurídica, os requisitos legais essenciais, a coerência da
   fundamentação e a aderência às referências fornecidas.
4. Não invente leis, artigos, súmulas, ementas, números de processos, datas ou
   entendimentos jurisprudenciais.
5. Não presuma que uma referência citada pela resposta é verdadeira ou pertinente
   sem evidência suficiente.
6. Quando a pergunta exigir quantidade específica de julgados, avalie o atendimento
   desse requisito e a possibilidade de verificar a autenticidade e a pertinência
   das decisões.
7. Não confunda sentimento processual com sentimento emocional. Quando solicitado,
   o sentimento representa a orientação favorável ou desfavorável à pretensão
   jurídica, conforme o gabarito.
8. Não penalize automaticamente diferenças de redação, organização ou terminologia
   quando o conteúdo jurídico estiver correto.
9. Se as referências forem insuficientes para confirmar uma afirmação, registre a
   limitação e não presuma sua veracidade.
10. Não atribua superioridade à abordagem RAG apenas por citar fontes ou referências.

# CRITÉRIOS (notas independentes de 0 a 5)

- nota_fidelidade: correspondência entre a resposta e os elementos essenciais da
  RESPOSTA ESPERADA (conclusão e condições jurídicas apresentadas).
- nota_precisao_gabarito: correção da interpretação jurídica, observância dos
  requisitos legais relevantes e compatibilidade com as REFERÊNCIAS. Uma conclusão
  aparentemente correta NÃO recebe nota máxima se o fundamento jurídico for
  incompatível com as referências.
- nota_completude: atendimento de TODOS os elementos exigidos pela pergunta e pelo
  gabarito (requisitos, explicações, QUANTIDADE de julgados solicitada e
  classificação do sentimento processual, quando aplicáveis).

Escala comum:
0 = ausente, integralmente incorreto ou incompatível com o requisito avaliado.
1 = atendimento mínimo, com erros graves.
2 = atendimento parcial, com lacunas ou incorreções relevantes.
3 = atendimento satisfatório, mas com limitações identificáveis.
4 = atendimento consistente, com pequenas omissões ou imprecisões.
5 = atendimento integral e consistente com os critérios estabelecidos.
A nota reflete o critério avaliado, não uma impressão geral da resposta.

# ERROS CRÍTICOS

Compare a resposta com o campo INACEITÁVEL. Ao identificar uma violação, descreva
objetivamente o trecho ou conteúdo problemático e sua incompatibilidade com o
gabarito, registrando cada ocorrência em "erros_criticos". Erros críticos devem
refletir-se nas notas de precisão, fidelidade ou completude, conforme a natureza
do erro. Uma boa nota em outros critérios NÃO compensa um erro jurídico crítico.

# SENTIMENTO PROCESSUAL

Quando a pergunta solicitar o sentimento ligado à probabilidade de sucesso/derrota,
classifique em:
- positivo: orientação favorável à pretensão da parte autora.
- negativo: orientação desfavorável à pretensão da parte autora.
- neutro ou inconclusivo: sem elementos suficientes para classificação fundamentada.
- nao_aplicavel: a pergunta não solicita sentimento.
Não interprete essas categorias como probabilidades estatísticas reais de vitória.
Verifique se a classificação da resposta corresponde à orientação do gabarito.

# RASTREABILIDADE DE JULGADOS

Distinga afirmações sustentadas pelas referências de afirmações não verificáveis.
NÃO declare que julgados foram verificados sem evidências suficientes para confirmar
sua autenticidade.

# REFERÊNCIA ÀS RESPOSTAS

Na justificativa, use SEMPRE "a resposta da LLM Padrão" e "a resposta da LLM com
RAG". NUNCA use "resposta A", "resposta B", "A" ou "B".

# FORMATO DE SAÍDA

Responda EXCLUSIVAMENTE com um objeto JSON válido, sem Markdown nem comentários,
no formato (os mesmos campos para cada abordagem):
{
  "avaliacao_llm_padrao": {
    "nota_fidelidade": <0-5>,
    "nota_precisao_gabarito": <0-5>,
    "nota_completude": <0-5>,
    "sentimento_identificado": "<positivo|negativo|neutro|inconclusivo|nao_aplicavel>",
    "sentimento_aderente_gabarito": <true|false|null>,
    "erros_criticos": ["<descrição objetiva>", "..."],
    "julgados_solicitados": <inteiro>,
    "julgados_apresentados": <inteiro>,
    "julgados_verificaveis": <inteiro>,
    "justificativa_juiz": "<texto conciso>"
  },
  "avaliacao_llm_rag": {
    "nota_fidelidade": <0-5>,
    "nota_precisao_gabarito": <0-5>,
    "nota_completude": <0-5>,
    "sentimento_identificado": "<positivo|negativo|neutro|inconclusivo|nao_aplicavel>",
    "sentimento_aderente_gabarito": <true|false|null>,
    "erros_criticos": ["<descrição objetiva>", "..."],
    "julgados_solicitados": <inteiro>,
    "julgados_apresentados": <inteiro>,
    "julgados_verificaveis": <inteiro>,
    "justificativa_juiz": "<texto conciso>"
  }
}

Se não houver erros críticos, use lista vazia. Use "julgados_verificaveis" apenas
para decisões cuja autenticidade e pertinência puderem ser confirmadas pelas
evidências disponíveis; na dúvida, use 0. A avaliação deve basear-se SOMENTE nos
dados fornecidos, sem criar resultados, referências ou fatos jurídicos inexistentes.
"""


def _avaliacao_vazia(justificativa: str) -> dict[str, Any]:
    return {
        "nota_fidelidade": 0,
        "nota_precisao_gabarito": 0,
        "nota_completude": 0,
        "sentimento_identificado": "inconclusivo",
        "sentimento_aderente_gabarito": None,
        "erros_criticos": [justificativa],
        "julgados_solicitados": 0,
        "julgados_apresentados": 0,
        "julgados_verificaveis": 0,
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

            # response_format força a saída a ser um objeto JSON.
            model_kwargs: dict[str, Any] = {
                "response_format": {"type": "json_object"}
            }
            # reasoning_effort reduz os tokens gastos no raciocínio interno dos
            # modelos gpt-oss. Sem isso, o modelo pode consumir todo o orçamento
            # de max_tokens no raciocínio e não sobrar espaço para emitir o JSON,
            # resultando em geração vazia e erro 400 (json_validate_failed).
            reasoning_effort = self.params.get("reasoning_effort")
            if reasoning_effort:
                model_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

            self._llm = ChatGroq(
                model=self.modelo,
                temperature=0.0,
                api_key=self.config.groq_api_key,
                max_retries=0,
                # Teto de saída: precisa acomodar o raciocínio interno dos modelos
                # gpt-oss MAIS o JSON com as duas avaliações. Um teto muito baixo
                # trunca o JSON e dispara erro 400 na Groq.
                max_tokens=self.params.get("max_tokens_juiz", 2000),
                model_kwargs=model_kwargs,
            )
        return self._llm

    def _get_llm_json_livre(self):
        """
        LLM do Juiz SEM response_format (json_object), usado como fallback quando
        a Groq rejeita o modo JSON estrito (erro 400 json_validate_failed). O JSON
        é extraído da saída em texto pelo parser tolerante _parse_json.
        """
        from langchain_groq import ChatGroq

        model_kwargs: dict[str, Any] = {}
        reasoning_effort = self.params.get("reasoning_effort")
        if reasoning_effort:
            model_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        return ChatGroq(
            model=self.modelo,
            temperature=0.0,
            api_key=self.config.groq_api_key,
            max_retries=0,
            max_tokens=self.params.get("max_tokens_juiz", 2000),
            model_kwargs=model_kwargs,
        )

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

        # Molde da mensagem do usuário, com placeholders para as respostas.
        # As respostas podem ser truncadas para caber no orçamento de tokens
        # (limite TPM da conta). O truncamento afeta APENAS o que vai ao Juiz;
        # as respostas completas continuam nos relatórios.
        molde = (
            "## PERGUNTA:\n{pergunta}\n\n"
            "## GABARITO OFICIAL:\n{gabarito}\n\n"
            "## RESPOSTA DA LLM PADRÃO (sem contexto):\n{rp}\n\n"
            "## RESPOSTA DA LLM COM RAG (com contexto jurídico):\n{rr}\n\n"
            "Avalie a resposta da LLM Padrão em \"avaliacao_llm_padrao\" e a "
            "resposta da LLM com RAG em \"avaliacao_llm_rag\".\n"
        )

        # Orçamento de tokens de ENTRADA (prompt) do Juiz. Deixamos margem para
        # a saída (max_tokens_juiz) dentro do limite TPM da conta.
        orcamento = int(self.params.get("juiz_orcamento_tokens_entrada", 6500))

        def montar_user_msg(rp: str, rr: str) -> str:
            return molde.format(pergunta=pergunta, gabarito=gabarito, rp=rp, rr=rr)

        def ajustar_para_orcamento(orc: int) -> tuple[str, bool]:
            """
            Trunca as respostas (proporcionalmente) para o prompt caber em `orc`
            tokens. Preserva pergunta/gabarito/system integralmente; o corte
            recai sobre as respostas, que são a maior fonte de tokens.
            Retorna (user_msg, houve_truncamento).
            """
            fixo = (
                _estimar_tokens(self.system_prompt)
                + _estimar_tokens(montar_user_msg("", ""))
            )
            disponivel = max(orc - fixo, 200)  # nunca abaixo de um mínimo
            tk_rp = _estimar_tokens(resposta_padrao)
            tk_rr = _estimar_tokens(resposta_rag)
            total_resp = tk_rp + tk_rr

            if total_resp <= disponivel:
                return montar_user_msg(resposta_padrao, resposta_rag), False

            # Divide o espaço disponível proporcionalmente ao tamanho de cada
            # resposta (a maior cede mais), garantindo um mínimo para cada uma.
            min_cada = min(300, disponivel // 2)
            cota_rp = max(min_cada, int(disponivel * (tk_rp / total_resp)))
            cota_rr = max(min_cada, disponivel - cota_rp)
            rp_t, t1 = _truncar_por_tokens(resposta_padrao, cota_rp)
            rr_t, t2 = _truncar_por_tokens(resposta_rag, cota_rr)
            return montar_user_msg(rp_t, rr_t), (t1 or t2)

        user_msg, _truncou = ajustar_para_orcamento(orcamento)

        llm = self._get_llm()

        def chamada():
            inicio = time.perf_counter()
            resp = llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_msg),
            ])
            decorrido = time.perf_counter() - inicio
            return resp.content, decorrido

        try:
            content, tempo = com_backoff(
                chamada,
                max_retries=self.rate.get("max_retries", 6),
                base_delay_s=self.rate.get("base_delay_s", 2.0),
                max_delay_s=self.rate.get("max_delay_s", 60.0),
                on_wait=on_wait,
            )
        except Exception as exc:  # noqa: BLE001
            # Erro 413 (request too large): o prompt ainda estourou o limite.
            # Reduz o orçamento agressivamente e tenta mais uma vez.
            if _e_request_too_large(exc):
                orc_reduzido = int(orcamento * 0.6)
                user_msg, _ = ajustar_para_orcamento(orc_reduzido)

                def chamada2():
                    inicio = time.perf_counter()
                    resp = self._get_llm().invoke([
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=user_msg),
                    ])
                    return resp.content, time.perf_counter() - inicio

                content, tempo = com_backoff(
                    chamada2,
                    max_retries=self.rate.get("max_retries", 6),
                    base_delay_s=self.rate.get("base_delay_s", 2.0),
                    max_delay_s=self.rate.get("max_delay_s", 60.0),
                    on_wait=on_wait,
                )
            elif _e_json_validate_failed(exc):
                # A Groq rejeitou o modo JSON estrito (geração vazia). Refaz a
                # chamada SEM response_format e extrai o JSON do texto gerado,
                # com instrução explícita para responder só com o objeto JSON.
                llm_livre = self._get_llm_json_livre()
                reforco = (
                    user_msg
                    + "\n\nIMPORTANTE: responda EXCLUSIVAMENTE com o objeto JSON "
                    "no formato especificado, sem nenhum texto antes ou depois."
                )

                def chamada_livre():
                    inicio = time.perf_counter()
                    resp = llm_livre.invoke([
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=reforco),
                    ])
                    return resp.content, time.perf_counter() - inicio

                content, tempo = com_backoff(
                    chamada_livre,
                    max_retries=self.rate.get("max_retries", 6),
                    base_delay_s=self.rate.get("base_delay_s", 2.0),
                    max_delay_s=self.rate.get("max_delay_s", 60.0),
                    on_wait=on_wait,
                )
            else:
                raise

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

            def _n_erros(av: dict) -> int:
                erros = av.get("erros_criticos") or []
                return len(erros) if isinstance(erros, list) else 0

            linhas_casos.append(
                f"- Pergunta {r.get('id_pergunta')}: "
                f"Padrão [fid={av_p.get('nota_fidelidade')}, "
                f"prec={av_p.get('nota_precisao_gabarito')}, "
                f"compl={av_p.get('nota_completude')}, "
                f"erros_críticos={_n_erros(av_p)}] | "
                f"RAG [fid={av_r.get('nota_fidelidade')}, "
                f"prec={av_r.get('nota_precisao_gabarito')}, "
                f"compl={av_r.get('nota_completude')}, "
                f"erros_críticos={_n_erros(av_r)}]"
            )
        resumo_casos = "\n".join(linhas_casos)

        resumo_metricas = f"""Médias globais (escala 0 a 5):
- Fidelidade: Padrão={metricas.get('nota_fidelidade_media_padrao')} | RAG={metricas.get('nota_fidelidade_media_rag')}
- Precisão vs. gabarito: Padrão={metricas.get('nota_precisao_media_padrao')} | RAG={metricas.get('nota_precisao_media_rag')}
- Completude: Padrão={metricas.get('nota_completude_media_padrao')} | RAG={metricas.get('nota_completude_media_rag')}
- Taxa de erros críticos: Padrão={metricas.get('taxa_erros_criticos_padrao_pct')}% | RAG={metricas.get('taxa_erros_criticos_rag_pct')}%
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

Comece a resposta com UMA linha contendo apenas a palavra "VENCEDORA:" seguida
de exatamente uma destas opções (sem colchetes, sem outros caracteres): RAG, Padrão
ou Empate. Exemplos válidos de primeira linha: "VENCEDORA: RAG" ou "VENCEDORA: Padrão".

Em seguida, em nova linha, escreva o texto do parecer."""

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

        texto = (content or "").strip()

        # Fallback: decide a vencedora pelas métricas (precisão; desempata por
        # fidelidade e completude). Usado quando o modelo não devolve um valor
        # válido na linha "VENCEDORA:".
        def _vencedora_por_metricas() -> str:
            def _soma(suf: str) -> float:
                return (
                    (metricas.get(f"nota_precisao_media_{suf}") or 0)
                    + (metricas.get(f"nota_fidelidade_media_{suf}") or 0)
                    + (metricas.get(f"nota_completude_media_{suf}") or 0)
                )
            s_padrao, s_rag = _soma("padrao"), _soma("rag")
            if abs(s_rag - s_padrao) < 1e-9:
                return "Empate"
            return "RAG" if s_rag > s_padrao else "Padrão"

        def _canonizar(valor: str) -> str | None:
            """Mapeia o texto bruto para RAG | Padrão | Empate, ou None."""
            v = (valor or "").strip().strip(".<>[]()").lower()
            if v in ("rag", "llm rag", "llm com rag", "com rag"):
                return "RAG"
            if v in ("padrão", "padrao", "llm padrão", "llm padrao"):
                return "Padrão"
            if v in ("empate", "empatado", "equivalente"):
                return "Empate"
            return None

        # Extrai a abordagem vencedora da primeira linha "VENCEDORA: ..."
        abordagem: str | None = None
        linhas = texto.splitlines()
        linhas_parecer = linhas
        for i, linha in enumerate(linhas):
            if linha.strip().upper().startswith("VENCEDORA:"):
                abordagem = _canonizar(linha.split(":", 1)[1])
                # Remove a linha de metadado do texto exibido do parecer.
                linhas_parecer = linhas[:i] + linhas[i + 1:]
                break

        if abordagem is None:
            abordagem = _vencedora_por_metricas()

        parecer_texto = "\n".join(linhas_parecer).strip()

        return {
            "abordagem_vencedora": abordagem,
            "parecer_texto": parecer_texto,
            "tempo_parecer_juiz_s": round(tempo, 4),
        }

    @staticmethod
    def _normalizar_avaliacao(av: Any) -> dict[str, Any]:
        """
        Garante que uma avaliação individual tenha todos os campos do contrato,
        com tipos corretos, mesmo que o modelo omita ou troque o tipo de algum.
        """
        if not isinstance(av, dict):
            return _avaliacao_vazia("Avaliação ausente ou em formato inválido.")

        def _nota(valor: Any) -> int:
            try:
                return max(0, min(5, int(round(float(valor)))))
            except (TypeError, ValueError):
                return 0

        def _inteiro(valor: Any) -> int:
            try:
                return max(0, int(round(float(valor))))
            except (TypeError, ValueError):
                return 0

        # erros_criticos: aceita lista, string única ou ausência.
        erros = av.get("erros_criticos", [])
        if isinstance(erros, str):
            erros = [erros] if erros.strip() else []
        elif isinstance(erros, list):
            erros = [str(e) for e in erros if str(e).strip()]
        else:
            erros = []

        sentimento = str(av.get("sentimento_identificado", "inconclusivo") or
                         "inconclusivo").strip().lower()
        if sentimento not in {"positivo", "negativo", "neutro",
                              "inconclusivo", "nao_aplicavel"}:
            sentimento = "inconclusivo"

        aderente = av.get("sentimento_aderente_gabarito", None)
        if not isinstance(aderente, bool):
            aderente = None

        return {
            "nota_fidelidade": _nota(av.get("nota_fidelidade")),
            "nota_precisao_gabarito": _nota(av.get("nota_precisao_gabarito")),
            "nota_completude": _nota(av.get("nota_completude")),
            "sentimento_identificado": sentimento,
            "sentimento_aderente_gabarito": aderente,
            "erros_criticos": erros,
            "julgados_solicitados": _inteiro(av.get("julgados_solicitados")),
            "julgados_apresentados": _inteiro(av.get("julgados_apresentados")),
            "julgados_verificaveis": _inteiro(av.get("julgados_verificaveis")),
            "justificativa_juiz": str(av.get("justificativa_juiz", "") or ""),
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

        if not isinstance(data, dict):
            data = {}

        if "avaliacao_llm_padrao" not in data:
            data["avaliacao_llm_padrao"] = _avaliacao_vazia(
                "Falha ao parsear avaliação do juiz."
            )
        else:
            data["avaliacao_llm_padrao"] = Judge._normalizar_avaliacao(
                data["avaliacao_llm_padrao"]
            )

        if "avaliacao_llm_rag" not in data:
            data["avaliacao_llm_rag"] = _avaliacao_vazia(
                "Falha ao parsear avaliação do juiz."
            )
        else:
            data["avaliacao_llm_rag"] = Judge._normalizar_avaliacao(
                data["avaliacao_llm_rag"]
            )
        return data
