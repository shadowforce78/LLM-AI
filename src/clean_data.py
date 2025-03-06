import re
import json
import os
import os.path

# Déterminer le chemin racine du projet (en remontant depuis le répertoire src)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Remonte d'un niveau depuis src

# Construire les chemins complets
input_dir = os.path.join(project_root, "data", "extracted")
output_dir = os.path.join(project_root, "data", "cleaned")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)


def clean_text(text):
    text = re.sub(r"\{\{.*?\}\}", "", text)  # Supprime les templates
    text = re.sub(r"\[\[Catégorie:.*?\]\]", "", text)  # Supprime les catégories
    text = re.sub(r"==.*?==", "", text)  # Supprime les titres de sections
    text = text.lower().strip()  # Minuscule + trim
    return text


# Lire et nettoyer les fichiers extraits en parcourant récursivement les dossiers
for root, dirs, files in os.walk(input_dir):
    for file in files:

        # Chemin du fichier source
        input_file_path = os.path.join(root, file)

        # Construire le chemin de sortie correspondant
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Obtenir le nom de base du fichier sans extension et ajouter l'extension .json
        filename_base = os.path.splitext(file)[0]
        output_file_path = os.path.join(output_subdir, f"{filename_base}.json")

        # Traiter le fichier
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # WikiExtractor peut produire un fichier avec une entrée JSON par ligne
            cleaned_data = []
            for line in lines:
                if line.strip():
                    data = json.loads(line)
                    cleaned_data.append(
                        {
                            "title": data.get("title", ""),
                            "text": clean_text(data.get("text", "")),
                        }
                    )

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

            print(f"Traitement terminé: {input_file_path}")

        except Exception as e:
            print(f"Erreur lors du traitement de {input_file_path}: {e}")

print("✅ Nettoyage terminé !")
