@echo off
echo =============================
echo = TOKENISATION DES DONNEES  =
echo =============================

REM Activation de l'environnement Python
call conda activate llm-env 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo L'environnement conda "llm-env" n'existe pas.
    echo Tentative d'utilisation de l'environnement Python par défaut...
)

REM Change directory to project root to ensure proper path resolution
cd ..

REM Vérification de l'existence du fichier de données
if not exist "data\raw\wiki_dataset.json" (
    echo Les données brutes n'existent pas!
    echo Veuillez d'abord exécuter le script scrap.bat
    exit /b 1
)

REM Création des dossiers de sortie
if not exist "data\tokenized_dataset" mkdir "data\tokenized_dataset" 2>nul
if not exist "models\trained\trained_llm" mkdir "models\trained\trained_llm" 2>nul

echo Tokenisation des données en cours...
python src/tokenizer.py
if %ERRORLEVEL% NEQ 0 (
    echo Une erreur s'est produite lors de la tokenisation.
    exit /b 1
)

echo Tokenisation terminée avec succès!
pause
