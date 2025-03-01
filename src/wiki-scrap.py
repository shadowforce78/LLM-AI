import wikipediaapi
import json
from tqdm import tqdm
import os
import sys
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import threading
import random


# Set up the path to ensure consistent operation regardless of execution directory
def get_project_root():
    """Determine the project root directory based on execution context"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If running from src directory
    if os.path.basename(script_dir) == "src":
        return os.path.dirname(script_dir)

    # If running from scripts directory or elsewhere
    return os.path.abspath(os.path.join(script_dir, ".."))


# Get project root and set output path
project_root = get_project_root()
output_path = os.path.join(project_root, "data", "raw", "wiki_dataset.json")


# Cette fonction sera appelée par chaque processus, donc elle doit créer sa propre instance de WikipediaAPI
def create_wiki_instance():
    user_agent = "WikiScraperBot/1.0 (https://github.com/shadowforce78/wiki-scraper; planque.adam@email.com)"
    return wikipediaapi.Wikipedia(user_agent=user_agent, language="fr")


# Variables globales qui seront définies dans le bloc principal
shared_results = None
results_lock = None
seen_titles = None


def init_worker():
    """Initilialise l'environnement pour chaque worker"""
    # Création d'une nouvelle instance Wikipedia pour chaque processus
    global wiki_wiki
    wiki_wiki = create_wiki_instance()


def fetch_article_and_links(
    article_title, category, shared_seen_titles, depth=2, max_pages=25
):
    """Récupère un article et ses liens avec plus de profondeur et plus de pages"""
    # Utiliser l'instance créée par init_worker ou en créer une nouvelle si nécessaire
    wiki_instance = create_wiki_instance()

    page = wiki_instance.page(article_title)
    if not page.exists() or article_title in shared_seen_titles:
        return []

    results = []

    # Ajouter l'article principal si pas déjà vu
    if article_title not in shared_seen_titles:
        shared_seen_titles[article_title] = 1
        results.append({"title": page.title, "text": page.text, "category": category})

    # Si nous devons explorer plus profondément, récupérer les liens
    if depth > 0:
        linked_titles = []
        try:
            # Obtenir la liste des liens (limités à max_pages pour éviter la surcharge)
            linked_pages = list(page.links.items())[:max_pages]
            for link_title, link_page in linked_pages:
                if link_page.namespace == 0 and link_title not in shared_seen_titles:
                    shared_seen_titles[link_title] = 1
                    linked_titles.append(link_title)
        except Exception as e:
            print(
                f"Erreur lors de la récupération des liens pour {article_title}: {str(e)}"
            )

        # Si depth > 1 et qu'on a des liens, les ajouter pour exploration future
        if depth > 1 and linked_titles:
            return results, linked_titles

    return results


def process_category_single_thread(category, articles):
    """Version single-thread du traitement des catégories pour Windows"""
    print(f"\nTraitement de la catégorie : {category} (mode mono-thread)")
    results = []
    seen = {}

    # Traiter les articles principaux séquentiellement
    for title in tqdm(articles, desc="Articles principaux"):
        wiki_instance = create_wiki_instance()
        page = wiki_instance.page(title)

        if page.exists() and title not in seen:
            seen[title] = 1
            results.append(
                {"title": page.title, "text": page.text, "category": category}
            )

            # Récupérer quelques liens directs
            linked_pages_count = 0
            for link_title, link_page in list(page.links.items())[
                :10
            ]:  # Limité à 10 liens
                if linked_pages_count >= 10:
                    break
                if (
                    link_page.exists()
                    and link_page.namespace == 0
                    and link_title not in seen
                ):
                    seen[link_title] = 1
                    results.append(
                        {
                            "title": link_title,
                            "text": link_page.text,
                            "category": f"{category}_linked",
                        }
                    )
                    linked_pages_count += 1

    return results


def process_category_parallel(category, articles, manager_dict):
    """Version multi-thread mais compatible Windows du traitement des catégories avec plus de profondeur"""
    print(f"\nTraitement parallèle de la catégorie : {category} (mode ThreadPool)")
    results = []
    seen = manager_dict  # Dict partagé entre les threads

    # Fonction à exécuter pour chaque article
    def process_article(title):
        article_results = []
        wiki_instance = create_wiki_instance()
        page = wiki_instance.page(title)

        if not page.exists():
            return []

        # Ajouter l'article principal s'il n'est pas déjà vu
        with lock:
            if title not in seen:
                seen[title] = 1
                article_results.append({"title": page.title, "text": page.text, "category": category})

        # Récupérer plus de liens (25 au lieu de 10)
        linked_pages = []
        try:
            for link_title, link_page in list(page.links.items())[:25]:  # Augmenté à 25 liens
                with lock:
                    if link_page.exists() and link_page.namespace == 0 and link_title not in seen:
                        seen[link_title] = 1
                        linked_pages.append(link_title)
        except Exception as e:
            print(f"Erreur lors de la récupération des liens pour {title}: {str(e)}")

        # Traiter plus de liens récupérés, avec une exploration aléatoire supplémentaire
        for link_title in linked_pages:
            try:
                link_page = wiki_instance.page(link_title)
                article_results.append({
                    "title": link_title,
                    "text": link_page.text,
                    "category": f"{category}_linked",
                })
                
                # Explorer un niveau supplémentaire pour 20% des liens (exploration aléatoire)
                if random.random() < 0.2:  # Probabilité de 20%
                    try:
                        # Prendre 3 liens aléatoires de second niveau
                        secondary_links = list(link_page.links.items())
                        random.shuffle(secondary_links)
                        secondary_links = secondary_links[:3]
                        
                        for sec_link_title, sec_link_page in secondary_links:
                            with lock:
                                if sec_link_page.exists() and sec_link_page.namespace == 0 and sec_link_title not in seen:
                                    seen[sec_link_title] = 1
                                    try:
                                        sec_page = wiki_instance.page(sec_link_title)
                                        article_results.append({
                                            "title": sec_link_title,
                                            "text": sec_page.text,
                                            "category": f"{category}_secondary",
                                        })
                                    except Exception as e:
                                        print(f"Erreur exploration secondaire: {str(e)}")
                    except Exception as e:
                        pass  # Ignorer les erreurs d'exploration secondaire
                
            except Exception as e:
                print(f"Erreur lors du traitement du lien {link_title}: {str(e)}")

        return article_results

    # Utiliser ThreadPoolExecutor au lieu de ProcessPoolExecutor pour Windows
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()  # Pour protéger l'accès au dictionnaire partagé

    with ThreadPoolExecutor(max_workers=12) as executor:  # Plus de workers
        future_to_article = {executor.submit(process_article, title): title for title in articles}

        for future in tqdm(as_completed(future_to_article), total=len(future_to_article), desc="Articles"):
            try:
                article_results = future.result()
                results.extend(article_results)
            except Exception as e:
                title = future_to_article[future]
                print(f"Erreur lors du traitement de {title}: {str(e)}")

    return results


# Catégories principales et leurs articles associés
categories = {
    "Base": [
        "France",
        "Paris",
        "Lyon",
        "Marseille",
        "Bordeaux",
        "Toulouse",
        "Strasbourg",
        "Géographie_de_la_France",
        "Histoire_de_France",
        "Culture_française",
        "Économie_de_la_France", "Départements_français", "Régions_françaises",
        "Villes_de_France", "Monuments_français", "Politique_française",
        "Éducation_en_France", "Transport_en_France", "Gastronomie_française",
        "Arts_en_France", "Personnalités_françaises",
    ],
    "Tech": [
        "Intelligence_artificielle",
        "Apprentissage_automatique",
        "Deep_learning",
        "Traitement_automatique_des_langues",
        "Apprentissage_profond",
        "Réseau_de_neurones_artificiels",
        "Transformateur_(apprentissage_profond)",
        "Big_data",
        "Science_des_données",
        "Informatique_quantique",
        "Internet", "Réseaux_informatiques", "Langage_de_programmation",
        "Systèmes_d'exploitation", "Cybersécurité", "Blockchain",
        "Cloud_computing", "Internet_des_objets", "Réalité_virtuelle",
        "Réalité_augmentée", "Robotique", "Algorithme", "Base_de_données",
    ],
    "Sciences": [
        "Biologie",
        "Chimie",
        "Physique",
        "Mathématiques",
        "Astronomie",
        "Géologie",
        "Science",
        "Médecine",
        "Psychologie",
        "Sociologie",
        "Génétique", "Écologie", "Neurosciences", "Paléontologie",
        "Zoologie", "Botanique", "Climatologie", "Microbiologie",
        "Archéologie", "Océanographie", "Agronomie", "Thermodynamique",
        "Relativité", "Mécanique_quantique", "Pharmacologie",
        "Immunologie", "Virologie", "Épidémiologie", "Ondes_gravitationnelles",
    ],
    "Culture": [
        "Musique",
        "Cinéma",
        "Littérature",
        "Peinture",
        "Sculpture",
        "Danse",
        "Théâtre",
        "Photographie",
        "Art",
        "Culture",
        "Architecture", "Bande_dessinée", "Jeu_vidéo", "Mode_(habillement)",
        "Sports", "Religion", "Philosophie", "Mythologie", "Opéra",
        "Artisanat", "Design", "Poésie", "Roman_(littérature)", "Festival",
        "Musée", "Patrimoine_culturel", "Gastronomie", "Tradition", "Folklore",
    ],
    "Géographie": [
        "Continent", "Pays", "Montagne", "Océan", "Mer", "Fleuve", "Rivière", 
        "Lac", "Île", "Climat", "Désert", "Forêt", "Capitale", "Ville",
        "Afrique", "Amérique_du_Nord", "Amérique_du_Sud", "Antarctique",
        "Asie", "Europe", "Océanie", "Catastrophe_naturelle", "Réchauffement_climatique",
    ],
    "Histoire": [
        "Antiquité", "Moyen_Âge", "Renaissance_(période_historique)", "Révolution_française",
        "Première_Guerre_mondiale", "Seconde_Guerre_mondiale", "Guerre_froide",
        "Civilisation", "Empire", "Monarchie", "République", "Révolution_industrielle",
        "Colonisation", "Décolonisation", "Archéologie", "Préhistoire",
        "Histoire_ancienne", "Histoire_médiévale", "Histoire_moderne",
        "Histoire_contemporaine", "Personnage_historique",
    ],
}


def main():
    """Fonction principale appelée uniquement si le script est exécuté directement"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()
    all_results = []

    # Utiliser la méthode appropriée selon la plateforme
    if sys.platform == "win32":
        # Pour Windows, utiliser un threading pool au lieu de multiprocessing
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        for category, articles in categories.items():
            category_results = process_category_parallel(
                category, articles, shared_dict
            )
            all_results.extend(category_results)

            # Sauvegarde intermédiaire après chaque catégorie
            intermediate_path = os.path.join(
                project_root, "data", "raw", f"wiki_dataset_{category}_temp.json"
            )
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(category_results, f, ensure_ascii=False, indent=2)
            print(
                f"Sauvegarde intermédiaire de la catégorie {category}: {len(category_results)} articles"
            )
    else:
        # Pour Linux/Unix, utiliser le multiprocessing comme prévu initialement
        with multiprocessing.Manager() as manager:
            shared_seen_titles = manager.dict()

            for category, articles in categories.items():
                print(
                    f"\nTraitement de la catégorie : {category} (mode multiprocessing)"
                )
                results = []

                with ProcessPoolExecutor(
                    max_workers=min(multiprocessing.cpu_count(), 8),
                    initializer=init_worker,
                ) as executor:
                    # Utiliser partial pour passer shared_seen_titles comme argument
                    func = partial(
                        fetch_article_and_links,
                        shared_seen_titles=shared_seen_titles,
                        depth=1,
                        max_pages=10,
                    )

                    future_to_article = {
                        executor.submit(func, title, category): title
                        for title in articles
                    }

                    linked_titles_for_depth2 = []
                    for future in tqdm(
                        as_completed(future_to_article),
                        total=len(future_to_article),
                        desc="Articles principaux",
                    ):
                        try:
                            article_result = future.result()
                            if article_result:
                                if isinstance(article_result, tuple):
                                    article_data, linked_titles = article_result
                                    results.extend(article_data)
                                    for linked_title in linked_titles:
                                        linked_titles_for_depth2.append(
                                            (linked_title, f"{category}_linked")
                                        )
                                else:
                                    results.extend(article_result)
                        except Exception as e:
                            title = future_to_article[future]
                            print(f"Erreur lors du traitement de {title}: {str(e)}")

                all_results.extend(results)

                # Sauvegarde intermédiaire
                intermediate_path = os.path.join(
                    project_root, "data", "raw", f"wiki_dataset_{category}_temp.json"
                )
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(
                    f"Sauvegarde intermédiaire de la catégorie {category}: {len(results)} articles"
                )

    # Sauvegarde finale dans data/raw
    print(f"Saving data to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    elapsed_time = time.time() - start_time
    print(
        f"\nExtraction terminée en {elapsed_time:.2f} secondes! {len(all_results)} articles extraits"
    )
    print(
        f"Vitesse moyenne: {len(all_results) / elapsed_time:.2f} articles par seconde"
    )

    # Afficher les statistiques du dataset avant de terminer
    all_categories = {}
    for result in all_results:
        category = result["category"]
        if category in all_categories:
            all_categories[category] += 1
        else:
            all_categories[category] = 1
    
    print("\n📊 STATISTIQUES DU DATASET:")
    total = sum(all_categories.values())
    print(f"Total: {total} articles")
    
    for cat, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat}: {count} articles ({count/total*100:.1f}%)")


if __name__ == "__main__":
    # Ajouter freeze_support pour Windows
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
