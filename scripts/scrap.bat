@echo off
echo ================================
echo = COLLECTE DE DONNEES WIKIPEDIA =
echo ================================

REM Activation de l'environnement Python
call conda activate llm-env 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo L'environnement conda "llm-env" n'existe pas.
    echo Tentative d'utilisation de l'environnement Python par défaut...
)

REM Création du dossier de sortie
if not exist "..\data\raw" (
    mkdir "..\data\raw" 2>nul
)

echo Collecte des données Wikipedia en cours...
python ../src/wiki-scrap.py
if %ERRORLEVEL% NEQ 0 (
    echo Une erreur s'est produite lors de la collecte des données.
    exit /b 1
)

echo Collecte terminée avec succès!
echo Vous pouvez maintenant exécuter tokenizer.bat
pause
