@echo off
echo ===================================
echo = LANCEMENT DE L'ENTRAINEMENT LLM =
echo ===================================

REM Activation de l'environnement Python
call conda activate llm-env 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo L'environnement conda "llm-env" n'existe pas.
    echo Tentative d'utilisation de l'environnement Python par défaut...
)

REM Set PYTHONPATH to include project root for proper imports
cd ..
set PYTHONPATH=%CD%
echo Setting PYTHONPATH to: %PYTHONPATH%

REM Vérification de l'existence du fichier tokenized_dataset
if not exist "data\tokenized_dataset" (
    echo Les données tokenisées n'existent pas!
    echo Veuillez d'abord exécuter le script tokenizer.bat
    exit /b 1
)

REM Création des dossiers de sortie
if not exist "models\trained" mkdir "models\trained" 2>nul

echo Lancement de l'entrainement...
python src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Une erreur s'est produite lors de l'entrainement.
    exit /b 1
)

echo Entrainement terminé avec succès!
pause
