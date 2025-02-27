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

REM Vérification de l'existence du modèle entraîné (plusieurs emplacements possibles)
if not exist "..\models\trained" (
    if not exist "trained_llm" (
        echo Le modèle entraîné n'existe pas!
        echo Veuillez d'abord exécuter le processus complet (scrap, tokenize, train)
        exit /b 1
    )
)

echo Lancement de l'assistant IA...
python ../src/ai_assistant.py
if %ERRORLEVEL% NEQ 0 (
    echo Une erreur s'est produite lors du lancement de l'assistant.
    exit /b 1
)

pause
