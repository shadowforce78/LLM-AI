@echo off
echo ===================================
echo = LANCEMENT DE L'ASSISTANT IA LLM =
echo ===================================

REM Activation de l'environnement Python
call conda activate llm-env 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo L'environnement conda "llm-env" n'existe pas.
    echo Tentative d'utilisation de l'environnement Python par défaut...
)

REM Change directory to project root for proper path resolution
cd ..

REM Set PYTHONPATH to include project root for proper imports
set PYTHONPATH=%CD%

REM Affichage d'avertissement si le modèle n'est pas trouvé
if not exist "models\trained" (
    if not exist "trained_llm" (
        echo Avertissement: Modèle entraîné non trouvé.
        echo Un modèle pré-entraîné sera téléchargé depuis HuggingFace.
        echo Pour utiliser un modèle entraîné, exécutez d'abord le processus complet (scrap, tokenize, train)
    )
)

echo Lancement de l'assistant IA...
python src/ai_assistant.py
if %ERRORLEVEL% NEQ 0 (
    echo Une erreur s'est produite lors du lancement de l'assistant.
    exit /b 1
)

pause
