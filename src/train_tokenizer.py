from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os
import json

# 🔍 Déterminer le chemin racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 📂 Dossier contenant les textes nettoyés (chemins relatifs à la racine du projet)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "tokenized")
VOCAB_SIZE = 32_000  # Taille du vocabulaire à définir

# 📌 Charger les textes
print(f"📂 Chargement des textes depuis {DATA_DIR}...")
corpus = []
for file_name in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file_name)
    # Vérifier si c'est un fichier (et non un répertoire) et qu'il a une extension .json
    if os.path.isfile(file_path) and file_name.endswith('.json'):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for article in data:
                    corpus.append(article["text"])
            print(f"✓ Fichier {file_name} chargé avec succès")
        except PermissionError:
            print(f"⚠️ Erreur de permission: Impossible de lire {file_name}")
        except json.JSONDecodeError:
            print(f"⚠️ Erreur: {file_name} n'est pas un fichier JSON valide")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {file_name}: {str(e)}")

if not corpus:
    print("⚠️ Attention: Aucun texte n'a été chargé. Vérifiez le contenu du dossier.")
    exit(1)
else:
    print(f"✅ {len(corpus)} articles chargés avec succès")

# 🧩 Définition du tokenizer WordPiece
tokenizer = Tokenizer(models.BPE())  # Peut être changé en Unigram ou WordPiece
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 📚 Entraînement du tokenizer
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train_from_iterator(corpus, trainer)

# 📂 Sauvegarde du tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

print(f"✅ Tokenizer entraîné et sauvegardé dans {OUTPUT_DIR}/tokenizer.json")
