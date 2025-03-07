import os
import json
import random
import sys
import time
import concurrent.futures
from tokenizers import Tokenizer

# 🔍 Déterminer le chemin racine du projet (comme dans les autres scripts)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 📂 Chemins relatifs à la racine du projet
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "tokenized")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 📌 Charger le tokenizer entraîné
TOKENIZER_PATH = os.path.join(OUTPUT_PATH, "tokenizer.json")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# 📌 Initialiser les listes pour train et validation
train_data, val_data = [], []


# Fonction pour charger un fichier JSON
def load_json_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                articles = []
                if isinstance(data, list):
                    for article in data:
                        if "text" in article:
                            articles.append(article["text"])
                return articles, file_name, None
            except json.JSONDecodeError:
                return [], file_name, "Format JSON invalide"
    except PermissionError:
        return [], os.path.basename(file_path), "Erreur de permission"
    except Exception as e:
        return [], os.path.basename(file_path), str(e)


# Collecte des chemins de fichiers
print(f"📂 Recherche des fichiers JSON dans {DATA_PATH}...")
json_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))

print(f"🔍 {len(json_files)} fichiers JSON trouvés. Chargement en parallèle...")

# Chargement des fichiers en parallèle
corpus = []
file_count = 0
error_count = 0
start_time = time.time()

# Utiliser un ThreadPoolExecutor pour paralléliser le chargement des fichiers
with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 4)
) as executor:
    future_to_file = {
        executor.submit(load_json_file, file_path): file_path
        for file_path in json_files
    }

    for future in concurrent.futures.as_completed(future_to_file):
        articles, file_name, error = future.result()

        if error:
            print(f"⚠️ Erreur dans le fichier {file_name}: {error}")
            error_count += 1
        else:
            corpus.extend(articles)
            file_count += 1
            print(
                f"✅ Fichier {file_name} chargé avec succès ({len(articles)} articles)"
            )

loading_time = time.time() - start_time
print(f"⏱️ Temps de chargement: {loading_time:.2f} secondes")
print(f"\n📊 Statistiques du chargement:")
print(f"  - {len(corpus)} articles chargés")
print(f"  - {file_count} fichiers traités avec succès")
print(f"  - {error_count} erreurs rencontrées")

if not corpus:
    print(
        "⚠️ Aucun texte n'a été chargé. Vérification des permissions et du contenu des fichiers requise."
    )
    sys.exit(1)


# Fonction pour traiter un article (tokenisation et split train/val)
def process_article(args):
    i, article = args
    results = {"train": [], "val": [], "stats": {}}

    # Tokenisation
    tokens = tokenizer.encode(article).ids
    token_count = len(tokens)

    # Troncature pour éviter des séquences trop longues
    truncated = False
    if token_count > 512:
        tokens = tokens[:512]
        truncated = True

    # 🔀 Split en train (80%) et validation (20%)
    if random.random() < 0.8:
        results["train"] = tokens
    else:
        results["val"] = tokens

    results["stats"] = {"token_count": token_count, "truncated": truncated}

    return results


# 🔄 Parcourir les articles du corpus avec multithreading
print("\n🔄 Tokenisation et répartition du corpus en parallèle...")
start_time = time.time()
token_counts = []
truncated_count = 0
total_tokens = 0

# Préparation des arguments pour le traitement en parallèle
article_indices = [(i, article) for i, article in enumerate(corpus)]

# Traitement parallèle avec un rapport de progression
with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 2)
) as executor:
    futures = list(executor.map(process_article, article_indices))

    # Affichage de la progression
    for i, result in enumerate(futures):
        if i % 100 == 0 or i == len(corpus) - 1:
            progress = (i + 1) / len(corpus) * 100
            elapsed = time.time() - start_time
            estimated_total = elapsed / (i + 1) * len(corpus) if i > 0 else 0
            remaining = estimated_total - elapsed if i > 0 else 0
            print(
                f"  ⏳ {progress:.1f}% ({i+1}/{len(corpus)}) - Temps restant estimé: {remaining:.1f}s",
                end="\r",
            )

        # Collecte des résultats
        if result["train"]:
            train_data.append(result["train"])
        else:
            val_data.append(result["val"])

        # Collecte des statistiques
        token_count = result["stats"]["token_count"]
        token_counts.append(token_count)
        total_tokens += token_count

        if result["stats"]["truncated"]:
            truncated_count += 1

# Statistiques finales
print("\n\n📊 Statistiques de tokenisation:")
total_time = time.time() - start_time
avg_tokens = total_tokens / len(corpus) if corpus else 0
max_tokens = max(token_counts) if token_counts else 0
min_tokens = min(token_counts) if token_counts else 0

print(f"  - Temps de traitement: {total_time:.2f} secondes")
print(f"  - Articles traités: {len(corpus)}")
print(
    f"  - Articles tronqués: {truncated_count} ({truncated_count/len(corpus)*100:.1f}%)"
)
print(
    f"  - Tokens par article: {avg_tokens:.1f} en moyenne (min: {min_tokens}, max: {max_tokens})"
)
print(f"  - Total de tokens générés: {total_tokens}")

# 📂 Sauvegarde des fichiers
print("\n💾 Sauvegarde des données...")
with open(os.path.join(OUTPUT_PATH, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f)
with open(os.path.join(OUTPUT_PATH, "val.json"), "w", encoding="utf-8") as f:
    json.dump(val_data, f)

print(
    f"✅ Corpus généré : {len(train_data)} échantillons pour l'entraînement, {len(val_data)} pour la validation."
)
print(f"  - Taille train: {len(train_data)/len(corpus)*100:.1f}% du corpus")
print(f"  - Taille validation: {len(val_data)/len(corpus)*100:.1f}% du corpus")
print(f"  - Fichiers sauvegardés dans: {OUTPUT_PATH}")
