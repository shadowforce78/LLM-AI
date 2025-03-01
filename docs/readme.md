# LLM-AI French Language Model

Un modèle de langage en français basé sur GPT-2, entraîné sur un vaste corpus d'articles Wikipédia pour produire des réponses pertinentes et informatives.

## 🌟 Caractéristiques

- Modèle basé sur GPT-2 français (dbddv01/gpt2-french-small)
- Entraînement avancé sur un large corpus d'articles Wikipédia en français
- Traitement multiprocessing optimisé pour l'extraction de données
- Assistant IA simple qui s'appuie uniquement sur ses connaissances apprises
- Exploration approfondie de Wikipédia avec 6 catégories thématiques
- Support multi-plateforme (Windows, Linux, macOS)

## 🚀 Installation

```bash
# Cloner le projet
git clone https://github.com/votre-username/LLM-AI.git
cd LLM-AI

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Installer les dépendances
pip install transformers torch datasets wikipediaapi tqdm accelerate tensorboard
```

## 📁 Structure du Projet

```
LLM-AI/
├── data/
│   ├── raw/              # Données brutes extraites de Wikipédia
│   └── tokenized_dataset/ # Données tokenisées prêtes pour l'entraînement
├── src/
│   ├── wiki-scrap.py     # Extraction massive de données Wikipedia
│   ├── tokenizer.py      # Configuration et entraînement du tokenizer
│   ├── train.py          # Script d'entraînement optimisé
│   └── ai_assistant.py   # Interface utilisateur pour interagir avec le modèle
├── models/
│   ├── base/             # Configuration du modèle de base
│   └── trained/          # Modèle entraîné (résultat final)
├── scripts/              # Scripts utilitaires pour faciliter l'utilisation
└── docs/                 # Documentation du projet
```

## 🔧 Utilisation

Le projet propose une approche complète pour créer un modèle LLM personnalisé en français:

### 1. Extraction de données (Scraping)

Le script collecte automatiquement un vaste corpus d'articles Wikipédia à travers 6 catégories principales:
- Base (France, villes, géographie...)
- Tech (IA, informatique, réseaux...)
- Sciences (physique, biologie, mathématiques...)
- Culture (arts, musique, littérature...)
- Géographie (continents, pays, formations géographiques...)
- Histoire (périodes, événements, personnalités...)

```bash
python src/wiki-scrap.py
```

### 2. Préparation des données

Tokenisation des données extraites pour les rendre exploitables par le modèle:

```bash
python src/tokenizer.py
```

## 🔤 Le Tokenizer

Le tokenizer est un composant essentiel du pipeline de traitement du langage naturel. Il transforme le texte brut en tokens (unités lexicales) que le modèle peut comprendre.

### Caractéristiques du tokenizer

- **Base**: Adapté de GPT-2 français (dbddv01/gpt2-french-small)
- **Vocabulaire**: ~50K tokens optimisés pour le français
- **Tokens spéciaux**:
  - `<bos>`: Marqueur de début de séquence
  - `<eos>`: Marqueur de fin de séquence
  - `<pad>`: Token de padding pour uniformiser les longueurs
  - `<unk>`: Token pour les mots inconnus
- **Stratégie de padding**: Côté gauche (`padding_side="left"`)
- **Stratégie de troncature**: Côté gauche (`truncation_side="left"`)

### Traitement des données

Le tokenizer applique aux données brutes:
1. **Normalisation**: Standardisation du texte
2. **Segmentation**: Division en tokens pertinents
3. **Transformation Q&A**: Formatage des données en paires question-réponse
4. **Padding/Truncation**: Uniformisation des séquences à une longueur de 512 tokens

### Exemple de tokenisation

```python
# Exemple de tokenisation d'une question
question = "Quelle est la capitale de la France ?"
tokens = tokenizer.encode(question)
# => [1, 1158, 318, 287, 2255, 293, 287, 1567, 30, 13]

# Détokenisation
tokenizer.decode(tokens)
# => "<bos>Quelle est la capitale de la France ?<eos>"
```

Cette approche permet au modèle de traiter efficacement le français avec ses particularités linguistiques.

### 3. Entraînement du modèle

Processus d'entraînement optimisé avec suivi des métriques et sauvegarde des meilleurs checkpoints:

```bash
python src/train.py
```

### 4. Utilisation du modèle

L'assistant IA peut être utilisé via l'interface en ligne de commande:

```bash
python src/ai_assistant.py
# ou
scripts/use.bat  # Sur Windows
```

## 📊 Fonctionnalités avancées

### Extraction parallélisée des données

- **Multiprocessing** sur Linux/Unix et **Multithreading** sur Windows
- Exploration multi-niveaux des articles liés
- Échantillonnage aléatoire pour diversifier les sources
- Sauvegardes intermédiaires pour éviter la perte de données

### Entraînement optimisé

- Ajustement automatique des hyperparamètres selon la taille du dataset
- Monitoring avec TensorBoard
- Détection et sauvegarde des meilleurs modèles
- Early stopping intelligent avec période minimale garantie
- Visualisation ASCII de l'évolution de la perte en temps réel

### Assistant IA minimaliste

- Interface simple en ligne de commande
- Génération de réponses basées uniquement sur les connaissances apprises
- Support des questions factuelles, opinions et instructions

## 📚 Sources de données

L'assistant est entraîné sur les catégories d'articles Wikipédia suivantes:

- **Base**: France, Paris, Lyon, Marseille, Bordeaux, Toulouse, Strasbourg, et bien d'autres articles liés à la géographie française, l'histoire française et la culture française
- **Tech**: Intelligence artificielle, apprentissage automatique, deep learning, réseaux informatiques, cybersécurité, blockchain, cloud computing...
- **Sciences**: Biologie, chimie, physique, mathématiques, astronomie, génétique, neurosciences, médecine...
- **Culture**: Musique, cinéma, littérature, peinture, sculpture, danse, architecture, jeux vidéo...
- **Géographie**: Continents, pays, montagnes, océans, mers, fleuves, climats, déserts...
- **Histoire**: Antiquité, Moyen Âge, Renaissance, guerres mondiales, civilisations, empires...

Au total, plus de 100 thèmes principaux sont explorés, chacun générant de nombreux articles liés.

## 🔍 Performances

Le modèle est entraîné pour minimiser la perte tout en évitant le surapprentissage, avec des caractéristiques:

- **Vocabulaire**: Tokenizer français spécialisé avec ~50K tokens
- **Nombre de paramètres**: 124M (base GPT-2 small)
- **Longueur maximale de séquence**: 512 tokens en entraînement, extensible jusqu'à 1024 en inférence
- **Format d'entrée**: Paires de question-réponse formatées avec tokens spéciaux
- **Capacité de génération**: Textes cohérents et informatifs en français

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push sur la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ Limitations

- Le modèle est limité aux informations contenues dans son corpus d'entraînement
- L'assistant ne possède pas de connaissances sur les événements postérieurs à son entraînement
- Comme tous les modèles de langage, il peut parfois générer des informations incorrectes
- Les performances dépendent significativement du matériel utilisé (GPU recommandé)
