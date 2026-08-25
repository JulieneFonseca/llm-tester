@echo off
setlocal

REM ============================================================
REM  llm-tester - inicializador (interface Streamlit)
REM ============================================================

cd /d "%~dp0"

REM -- Verifica se o Python esta disponivel ---------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo Instale o Python 3.10+ em https://www.python.org/downloads/
    pause
    exit /b 1
)

REM -- Cria o ambiente virtual na primeira execucao ------------
if not exist ".venv\Scripts\activate.bat" (
    echo [setup] Criando ambiente virtual .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    echo [setup] Instalando dependencias...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call ".venv\Scripts\activate.bat"
)

REM -- Aviso sobre a chave de API ------------------------------
if not exist ".env" (
    echo [aviso] Arquivo .env nao encontrado.
    echo Copie .env.example para .env e defina sua GROQ_API_KEY.
    echo Chave gratuita: https://console.groq.com/keys
    echo.
)

REM -- Inicia a interface web ---------------------------------
echo [run] Iniciando llm-tester (Streamlit)...
streamlit run app.py

endlocal
