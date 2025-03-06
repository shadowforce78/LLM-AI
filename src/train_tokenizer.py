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

# 🧩 Définition du tokenizer - Utilisation du modèle BPE qui est plus flexible
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Configuration du pre-tokenizer pour un meilleur découpage initial
# Passage de add_prefix_space à True
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# Définition du décodeur correspondant au pre-tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
tokenizer.decoder = ByteLevelDecoder()

# Normalisation améliorée pour assurer le lowercasing et la suppression des accents
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# 📚 Entraînement du tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE, 
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"],
    min_frequency=2,
    show_progress=True
)
print(f"🔍 Entraînement du tokenizer (modèle BPE) avec {len(corpus)} textes...")
tokenizer.train_from_iterator(corpus, trainer)

# Test rapide de validation
test_text = "Bonjour, je suis un texte d'exemple."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids)
print("\n🧪 Test de validation:")
print(f"  Texte original: \"{test_text}\"")
print(f"  Tokens: {encoded.tokens}")
print(f"  Texte décodé: \"{decoded}\"")
print("  Note: Le tokenizer convertit le texte en lowercase et ajoute un espace initial.")

# Correction de la validation pour tenir compte de l'espace initial et du lowercase
expected = " " + test_text.lower()
is_match = decoded == expected
print(f"  Correspondance ajustée: {'✅ OK' if is_match else '❌ Différent'}")
if not is_match:
    print(f"  Texte attendu après ajustements: \"{expected}\"")

# 📂 Sauvegarde du tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

print(f"✅ Tokenizer entraîné et sauvegardé dans {OUTPUT_DIR}/tokenizer.json")
