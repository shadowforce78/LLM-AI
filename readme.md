# LLM-AI French Language Model

Un modèle de langage en français basé sur GPT-2, entraîné sur des articles Wikipédia.

## 🌟 Caractéristiques

- Modèle basé sur GPT-2 français (dbddv01/gpt2-french-small)
- Fine-tuning sur des articles Wikipédia en français
- Support des conversations et génération de texte
- Tokenizer optimisé pour le français
- Gestion automatique des données d'entraînement

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
pip install transformers torch datasets wikipediaapi tqdm accelerate
```

## 📁 Structure du Projet

```
LLM-AI/
├── wiki-scrap.py        # Extraction des données Wikipedia
├── tokenizer.py         # Configuration du tokenizer
├── modele_base.py       # Définition du modèle
├── train.py            # Script d'entraînement
├── test.py             # Script de test
└── trained_llm/        # Dossier du modèle entraîné
```

## 🔧 Utilisation

1. **Extraction des données** :
```bash
python wiki-scrap.py
```

2. **Préparation des tokens** :
```bash
python tokenizer.py
```

3. **Entraînement du modèle** :
```bash
python train.py
```

4. **Test du modèle** :
```bash
python test.py
```

## 📊 Exemple d'utilisation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained("trained_llm")
model = AutoModelForCausalLM.from_pretrained("trained_llm")

# Générer du texte
prompt = "Quelle est la capitale de la France ?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0])
print(response)
```

## 🎯 Paramètres d'entraînement

- Taille du modèle : 124M paramètres
- Epochs : 10
- Batch size : 4
- Learning rate : 5e-5
- Warmup steps : 500
- Beam search : 5 beams

## 📝 Notes

- Le modèle utilise GPT-2 comme architecture de base
- Les données d'entraînement sont extraites de Wikipédia
- Le tokenizer est optimisé pour le français
- Le modèle supporte les tokens spéciaux (BOS, EOS, PAD)

## 📚 Données

Les catégories d'articles Wikipédia utilisées :
- Base : Histoire, géographie et culture française
- Tech : IA, apprentissage automatique, deep learning

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push sur la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ Limitations

- Le modèle est entraîné sur un nombre limité d'articles
- Les performances peuvent varier selon la complexité des requêtes
- L'utilisation de CPU peut ralentir significativement l'inférence