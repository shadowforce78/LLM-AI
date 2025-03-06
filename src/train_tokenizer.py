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
print(f"📂 Chargement des textes depuis {DATA_DIR} et ses sous-dossiers...")
corpus = []
file_count = 0

# Fonction pour parcourir les dossiers de manière récursive
def load_files_from_dir(directory):
    global corpus, file_count
    
    for item_name in os.listdir(directory):
        item_path = os.path.join(directory, item_name)
        
        # Si c'est un dossier, on le parcourt récursivement
        if os.path.isdir(item_path):
            print(f"📁 Exploration du sous-dossier: {item_name}")
            load_files_from_dir(item_path)
        
        # Si c'est un fichier JSON, on le traite
        elif os.path.isfile(item_path) and item_name.endswith('.json'):
            try:
                with open(item_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for article in data:
                        corpus.append(article["text"])
                file_count += 1
                print(f"✓ Fichier {item_name} dans {os.path.basename(directory)} chargé avec succès")
            except PermissionError:
                print(f"⚠️ Erreur de permission: Impossible de lire {item_path}")
            except json.JSONDecodeError:
                print(f"⚠️ Erreur: {item_path} n'est pas un fichier JSON valide")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {item_path}: {str(e)}")

# Lancer le chargement des fichiers
load_files_from_dir(DATA_DIR)

if not corpus:
    print("⚠️ Attention: Aucun texte n'a été chargé. Vérifiez le contenu du dossier.")
    exit(1)
else:
    print(f"✅ {len(corpus)} articles chargés depuis {file_count} fichiers avec succès")

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
