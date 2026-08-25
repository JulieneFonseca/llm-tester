"""
Ponto de entrada da interface Streamlit do llm-tester.

Uso:
  streamlit run app.py
"""

import sys

sys.path.insert(0, "src")

from llm_tester.ui import run  # noqa: E402

if __name__ == "__main__":
    run()
else:
    # streamlit run app.py executa o módulo diretamente
    run()
