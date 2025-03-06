import os
import os.path

# Déterminer le chemin racine du projet (en remontant depuis le répertoire scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Remonte d'un niveau depuis scripts

# Construire les chemins complets
dump_file = os.path.join(project_root, "data", "raw", "frwiki-latest-pages-articles1.xml-p1p306134.bz2")
output_dir = os.path.join(project_root, "data", "extracted")

os.system(f"python3 -m wikiextractor.WikiExtractor {dump_file} -o {output_dir} --json --no-templates")
print(f"✅ Extraction terminée ! Résultats dans {output_dir}")
