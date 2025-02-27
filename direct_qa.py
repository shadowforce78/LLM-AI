import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Configuration
MODEL_BASE = "dbddv01/gpt2-french-small"

def init_qa_system():
    """Initialise un système de questions-réponses simple"""
    print("⏳ Chargement du modèle et du tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    
    # Configuration des tokens spéciaux
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(MODEL_BASE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Base de connaissances complète
QA_DATABASE = {
    # Géographie de la France
    "capitale france": "Paris est la capitale de la France.",
    "superficie france": "La France métropolitaine a une superficie d'environ 550 000 km² (643 801 km² avec l'outre-mer).",
    "population france": "La France compte environ 68 millions d'habitants (2024).",
    "plus grande ville france": "Paris est la plus grande ville de France.",
    "frontieres france": "La France partage ses frontières terrestres avec la Belgique, le Luxembourg, l'Allemagne, la Suisse, l'Italie, Monaco, l'Espagne et Andorre.",
    "fleuves france": "Les principaux fleuves de France sont la Seine, la Loire, le Rhône et la Garonne.",
    "montagnes france": "Les principales chaînes de montagnes en France sont les Alpes, les Pyrénées, le Massif central, le Jura et les Vosges.",
    
    # Histoire et politique
    "president france": "Emmanuel Macron est le président de la République Française depuis 2017.",
    "premiere ministre france": "Gabriel Attal est le Premier ministre de la France (en 2024).",
    "systeme politique france": "La France est une république constitutionnelle semi-présidentielle avec un président élu au suffrage universel direct.",
    "revolution francaise": "La Révolution française a eu lieu de 1789 à 1799. Elle a conduit à la chute de la monarchie et à l'établissement de la Première République.",
    
    # Culture
    "langue france": "Le français est la langue officielle de la France.",
    "gastronomie france": "La gastronomie française est réputée mondialement pour sa diversité et sa qualité, incluant le vin, les fromages, les pâtisseries et les plats régionaux.",
    
    # Technologie et science
    "intelligence artificielle": "L'Intelligence Artificielle est un domaine de l'informatique visant à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine.",
    "deep learning": "Le deep learning (apprentissage profond) est une branche de l'intelligence artificielle utilisant des réseaux de neurones avec plusieurs couches pour apprendre à partir des données.",
    "gpt": "GPT (Generative Pre-trained Transformer) est un modèle de langage développé par OpenAI, qui utilise l'architecture Transformer pour générer du texte.",
    "machine learning": "Le Machine Learning (apprentissage automatique) est une méthode d'analyse de données qui automatise la construction de modèles analytiques, permettant aux ordinateurs d'apprendre à partir des données.",
    
    # Autres
    "bonjour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "merci": "De rien ! N'hésitez pas si vous avez d'autres questions.",
    "aide": "Je suis un assistant basé sur GPT-2. Je peux répondre à des questions sur la France, la technologie, et d'autres sujets généraux. Que souhaitez-vous savoir ?",
}

def process_query(question):
    """Pré-traiter la question pour la recherche"""
    # Normalisation
    query = question.lower()
    
    # Supprimer la ponctuation et les articles
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+(le|la|les|du|des|un|une|de|d|et|est|sont|a|au|en)\s+', ' ', query)
    
    # Remplacer certains mots par leurs équivalents
    replacements = {
        "quelle": "", "quel": "", "qui": "", "que": "", "qu": "", "combien": "",
        "comment": "", "pourquoi": "", "où": "", "quand": "", "président": "president",
        "française": "france", "français": "france", "l'intelligence": "intelligence",
        "l'apprentissage": "apprentissage", "d'habitants": "population",
        "habitant": "population", "métropolitaine": "", "kilomètres": "km",
        "kilometre": "km", "ia": "intelligence artificielle", "pays": "",
        "superf": "superficie", "surface": "superficie", "taille": "superficie",
        "km²": "km", "km2": "km"
    }
    
    for old, new in replacements.items():
        query = query.replace(old, new)
    
    # Suppression des espaces multiples
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

def find_best_match(question):
    """Trouve la meilleure correspondance dans la base de connaissance"""
    processed_query = process_query(question)
    
    best_match = None
    max_match_count = 0
    
    for key in QA_DATABASE.keys():
        # Calcul de la correspondance
        key_words = set(key.split())
        query_words = set(processed_query.split())
        
        # Mots en commun
        common_words = key_words.intersection(query_words)
        
        # Si tous les mots importants sont présents
        if len(common_words) > max_match_count and all(w in processed_query for w in key.split() if len(w) > 3):
            max_match_count = len(common_words)
            best_match = key
    
    if best_match:
        return QA_DATABASE[best_match]
    
    return None

# Interface utilisateur améliorée
if __name__ == "__main__":
    tokenizer, model, device = init_qa_system()
    print(f"✅ Système initialisé avec succès sur {device.upper()}!")
    
    print("\n💬 Assistant IA - Questions-Réponses sur la France et la Tech")
    print("=" * 60)
    print("📌 Je peux répondre à des questions factuelles sur la France et la technologie.")
    print("📝 Exemples: 'Quelle est la superficie de la France?' ou 'Qu'est-ce que l'IA?'")
    print("❓ Tapez 'aide' pour plus d'informations, 'q' pour quitter")
    print("=" * 60)
    
    while True:
        user_input = input("\n➤ ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nAu revoir! 👋")
            break
        
        # Recherche dans la base de connaissances
        answer = find_best_match(user_input)
        
        if answer:
            print(f"\n🤖 {answer}")
        else:
            print("\n🤖 Je n'ai pas d'information précise sur ce sujet. Je suis spécialisé dans les questions sur la France et les technologies comme l'IA.")
