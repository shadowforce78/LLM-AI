import wikipediaapi
import json
from tqdm import tqdm

user_agent = "WikiScraperBot/1.0 (https://github.com/shadowforce78/wiki-scraper; planque.adam@email.com)"
wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="fr")

# Catégories principales et leurs articles associés
categories = {
    "Base": [
        "France", "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Strasbourg",
        "Géographie_de_la_France", "Histoire_de_France", "Culture_française"
    ],
    "Tech": [
        "Intelligence_artificielle", "Apprentissage_automatique", "Deep_learning",
        "Traitement_automatique_des_langues", "Apprentissage_profond",
        "Réseau_de_neurones_artificiels", "Transformateur_(apprentissage_profond)",
        "Big_data", "Science_des_données", "Informatique_quantique"
    ],
    "Sciences": [
        "Biologie", "Chimie", "Physique", "Mathématiques", "Astronomie", "Géologie",
        "Science", "Médecine", "Psychologie", "Sociologie"
    ],
    "Culture": [
        "Musique", "Cinéma", "Littérature", "Peinture", "Sculpture", "Danse", "Théâtre",
        "Photographie", "Art", "Culture"
    ]
}

def get_linked_pages(page, depth=1, max_pages=5):
    """Récupère récursivement les pages liées"""
    if depth <= 0 or not page.exists():
        return []
    
    linked_pages = []
    for link in list(page.links.keys())[:max_pages]:
        linked_page = wiki_wiki.page(link)
        if linked_page.exists() and linked_page.namespace == 0:
            linked_pages.append(linked_page)
            if depth > 1:
                linked_pages.extend(get_linked_pages(linked_page, depth-1, max_pages=2))
    return linked_pages

dataset = []
seen_titles = set()

for category, articles in categories.items():
    print(f"\nTraitement de la catégorie : {category}")
    for title in tqdm(articles, desc="Articles principaux"):
        page = wiki_wiki.page(title)
        if page.exists() and page.title not in seen_titles:
            dataset.append({"title": page.title, "text": page.text, "category": category})
            seen_titles.add(page.title)
            
            # Récupérer quelques pages liées
            linked_pages = get_linked_pages(page, depth=2)
            for linked_page in linked_pages:
                if linked_page.title not in seen_titles:
                    dataset.append({
                        "title": linked_page.title,
                        "text": linked_page.text,
                        "category": f"{category}_linked"
                    })
                    seen_titles.add(linked_page.title)

# Sauvegarde
with open("wiki_api_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\nExtraction terminée ! {len(dataset)} articles extraits")
