LLM-AI/
├── README.md                    # Documentation principale du projet
├── LICENSE                      # Licence MIT
├── requirements.txt             # Dépendances Python
│
├── config/                      # Configuration centralisée
│   ├── __init__.py              # Rend le répertoire importable
│   └── model_config.py          # Paramètres du modèle (ancien config_base.py)
│
├── data/                        # Gestion des données
│   ├── __init__.py
│   ├── scraper.py               # Extraction des données (ancien wiki-scrap.py)
│   └── processor.py             # Prétraitement des données
│
├── models/                      # Définition des modèles
│   ├── __init__.py
│   ├── tokenizer.py             # Tokenizer (ancien tokenizer.py)
│   └── gpt2_model.py            # Modèle GPT-2 (ancien modele_base.py)
│
├── training/                    # Logique d'entraînement
│   ├── __init__.py
│   └── trainer.py               # Script d'entraînement (ancien train.py)
│
├── inference/                   # Génération et inférence
│   ├── __init__.py
│   ├── generator.py             # Génération de texte
│   └── interactive_qa.py        # Interface QA interactive (ancien test.py)
│
├── utils/                       # Utilitaires partagés
│   ├── __init__.py
│   └── helpers.py               # Fonctions utilitaires
│
├── scripts/                     # Scripts autonomes
│   ├── train_model.py           # Script pour lancer l'entraînement
│   └── evaluate_model.py        # Évaluation des performances
│
└── trained_models/              # Dossier pour les modèles entraînés
    └── README.md                # Instructions pour les modèles entraînés
