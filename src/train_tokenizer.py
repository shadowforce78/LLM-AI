from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os
import json
import concurrent.futures
from transformers import PreTrainedTokenizerFast
import time

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
json_files = []


# Fonction pour charger un seul fichier JSON
def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            texts = [article["text"] for article in data]
        return texts, file_path, None
    except PermissionError:
        return [], file_path, "Erreur de permission: Impossible de lire le fichier"
    except json.JSONDecodeError:
        return [], file_path, "Erreur: Fichier JSON invalide"
    except Exception as e:
        return [], file_path, f"Erreur: {str(e)}"


# Fonction pour collecter tous les chemins de fichiers JSON
def collect_json_files(directory):
    global json_files

    for item_name in os.listdir(directory):
        item_path = os.path.join(directory, item_name)

        # Si c'est un dossier, on le parcourt récursivement
        if os.path.isdir(item_path):
            print(f"📁 Exploration du sous-dossier: {item_name}")
            collect_json_files(item_path)

        # Si c'est un fichier JSON, on l'ajoute à la liste
        elif os.path.isfile(item_path) and item_name.endswith(".json"):
            json_files.append(item_path)


# Collecter tous les fichiers JSON
collect_json_files(DATA_DIR)
print(f"🔍 {len(json_files)} fichiers JSON trouvés. Chargement en parallèle...")

# Charger les fichiers en parallèle
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 4)
) as executor:
    # Soumettre tous les fichiers pour traitement parallèle
    future_to_file = {
        executor.submit(load_json_file, file): file for file in json_files
    }

    # Collecter les résultats au fur et à mesure qu'ils sont terminés
    for future in concurrent.futures.as_completed(future_to_file):
        file_path = future_to_file[future]
        file_name = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))

        texts, _, error = future.result()

        if error:
            print(f"⚠️ {error}: {file_path}")
        else:
            corpus.extend(texts)
            file_count += 1
            print(f"✓ Fichier {file_name} dans {dir_name} chargé avec succès")

elapsed_time = time.time() - start_time
print(f"⏱️ Temps de chargement: {elapsed_time:.2f} secondes")

if not corpus:
    print("⚠️ Attention: Aucun texte n'a été chargé. Vérifiez le contenu du dossier.")
    exit(1)
else:
    print(f"✅ {len(corpus)} articles chargés depuis {file_count} fichiers avec succès")

# 🧩 Définition du tokenizer - Utilisation du modèle BPE qui est plus flexible
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Configuration du pre-tokenizer pour un meilleur découpage initial
# Passage de add_prefix_space à False pour éviter l'espace au début lors du décodage
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

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
    show_progress=True,
)
print(f"🔍 Entraînement du tokenizer (modèle BPE) avec {len(corpus)} textes...")
tokenizer.train_from_iterator(corpus, trainer)

# Test rapide de validation
test_text = "Bonjour, je suis un texte d'exemple."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids)
print("\n🧪 Test de validation:")
print(f'  Texte original: "{test_text}"')
print(f"  Tokens: {encoded.tokens}")
print(f'  Texte décodé: "{decoded}"')
print(
    "  Note: Le tokenizer convertit le texte en lowercase et peut modifier légèrement la tokenisation."
)

# Vérification plus précise tenant compte des transformations du tokenizer
lower_test = test_text.lower()
is_similar = lower_test in decoded or decoded in lower_test
print(f"  Correspondance approximative: {'✅ OK' if is_similar else '❌ Différent'}")

# Test plus complet montrant l'utilisation correcte pour l'entrainement de modèles
print("\n🔄 Test d'utilisation complète:")
sample_texts = [
    "Bonjour le monde!",
    "Comment ça va aujourd'hui?",
    "Le tokenizer est maintenant configuré.",
]
for text in sample_texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    print(f'  Original: "{text}"')
    print(f"  Encodé: {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
    print(f'  Décodé: "{decoded}"')
    print("  ---")

# 📂 Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Sauvegarde format `tokenizers`
tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
print(f"✅ Tokenizer sauvegardé dans {OUTPUT_DIR}/tokenizer.json")

# 🔹 Sauvegarde format Hugging Face
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
hf_tokenizer.save_pretrained(OUTPUT_DIR)
print(
    f"✅ Tokenizer Hugging Face sauvegardé dans {OUTPUT_DIR} (config.json & tokenizer.json)"
)
