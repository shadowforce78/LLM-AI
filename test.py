from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, pipeline
import torch
import re

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("trained_llm")
model = AutoModelForCausalLM.from_pretrained(
    "trained_llm",
    trust_remote_code=True,
    local_files_only=True
)
model.eval()
print("✅ Modèle et tokenizer chargés")

def format_prompt(text):
    """Formatte le prompt pour une meilleure génération"""
    # Contexte plus structuré pour guider la génération
    formatted = (
        f"{tokenizer.bos_token}### Question : {text}\n\n### Réponse : "
    )
    return formatted

# Base de connaissances enrichie
KNOWLEDGE_BASE = {
    "capitale": {
        "France": "Paris est la capitale de la France. C'est la plus grande ville du pays et son centre économique et culturel.",
        "default": "Je ne connais pas la capitale de ce pays."
    },
    "population": {
        "France": "La France compte environ 68 millions d'habitants (2024).",
        "Paris": "Paris compte environ 2,2 millions d'habitants, et son aire urbaine environ 12 millions d'habitants.",
        "default": "Je ne connais pas la population exacte."
    },
    "définition": {
        "IA": "L'Intelligence Artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de simuler l'intelligence humaine.",
        "default": "Je ne peux pas fournir une définition précise pour ce terme."
    }
}

def get_knowledge_base_answer(question):
    """Recherche une réponse dans la base de connaissances"""
    question = question.lower()
    
    # Règles de correspondance
    if any(word in question for word in ["habitant", "population"]):
        if "france" in question:
            return KNOWLEDGE_BASE["population"]["France"]
        if "paris" in question:
            return KNOWLEDGE_BASE["population"]["Paris"]
            
    if "capitale" in question and "france" in question:
        return KNOWLEDGE_BASE["capitale"]["France"]
        
    if ("qu'est" in question or "c'est quoi" in question) and "ia" in question:
        return KNOWLEDGE_BASE["définition"]["IA"]
        
    return None

# Option pour activer/désactiver la base de connaissances
USE_KNOWLEDGE_BASE = False  # Mettre à False pour utiliser uniquement le modèle

# Améliorer les paramètres de génération
def generate_response(prompt_text, max_new_tokens=100):
    """Génère une réponse avec des paramètres optimisés"""
    # Vérifier d'abord la base de connaissances si activée
    if USE_KNOWLEDGE_BASE:
        kb_answer = get_knowledge_base_answer(prompt_text)
        if kb_answer:
            return kb_answer

    # Essayer avec un prompt direct sans formatage complexe
    raw_prompt = f"{prompt_text}"
    
    inputs = tokenizer(
        raw_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,  # Réduit pour éviter les problèmes de contexte
    )
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=5,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,  # Plus de créativité
                top_k=50,        # Moins restrictif
                top_p=0.9,
                repetition_penalty=1.2,  # Moins restrictif
                no_repeat_ngram_size=3,
                num_beams=1,      # Greedy decoding pour plus de spontanéité
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Décodage basique
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated_text = generated_text.strip()
        
        # Vérifier si la réponse a du sens
        if len(generated_text) < 10 or "Notes et références" in generated_text:
            # Fallback à la base de connaissances si disponible
            fallback = get_fallback_answer(prompt_text)
            if fallback:
                return fallback
        
        return generated_text
    
    except Exception as e:
        print(f"Erreur de génération: {str(e)}")
        # Fallback à la base de connaissances
        fallback = get_fallback_answer(prompt_text)
        if fallback:
            return fallback
        return "Je ne peux pas répondre à cette question pour le moment."

def get_fallback_answer(question):
    """Fournit une réponse de secours pour les questions courantes"""
    question = question.lower()
    
    # Questions courantes et leurs réponses
    fallbacks = {
        "capitale france": "Paris est la capitale de la France.",
        "habitant france": "La France compte environ 68 millions d'habitants (2024).",
        "intelligence artificielle": "L'Intelligence Artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de simuler l'intelligence humaine.",
    }
    
    # Recherche de correspondance approximative
    for key, answer in fallbacks.items():
        if all(word in question for word in key.split()):
            return answer
            
    return None

def clean_response(text, original_prompt):
    """Nettoie et formate la réponse générée"""
    # Extraire la réponse après le marqueur
    try:
        response = re.split(r'###\s*Réponse\s*:', text)[-1].strip()
    except IndexError:
        response = text.replace(original_prompt, "").strip()
    
    # Nettoyage avancé
    response = re.sub(r'\s+', ' ', response)  # Normaliser les espaces
    response = re.sub(r'^\W+|\W+$', '', response)  # Nettoyer début/fin
    response = re.sub(r'Portail.*$', '', response)  # Supprimer les mentions "Portail"
    response = re.sub(r'###.*$', '', response)  # Supprimer les marqueurs restants
    response = re.sub(r'Articles connexes.*$', '', response)
    response = re.sub(r'Liens externes.*$', '', response)
    response = re.sub(r'ISBN.*$', '', response)
    response = re.sub(r'\([^)]*\)', '', response)  # Supprimer les parenthèses
    
    # Vérifications de qualité
    if len(response) < 10 or response.count(' ') < 2:
        return "Je ne peux pas générer une réponse cohérente à cette question."
    
    # Capitaliser la première lettre
    response = response[0].upper() + response[1:] if response else response
    
    return response

# Ajouter des questions d'exemple pour aider le modèle
EXAMPLE_PROMPTS = {
    "Quelle est la capitale de la France ?": "Paris est la capitale de la France.",
    "Qu'est-ce que l'IA ?": "L'Intelligence Artificielle (IA) est un domaine de l'informatique...",
}

def init_model_with_examples():
    """Initialise le modèle avec quelques exemples"""
    for q, a in EXAMPLE_PROMPTS.items():
        formatted = format_prompt(q) + a + tokenizer.eos_token
        _ = tokenizer(formatted, return_tensors="pt")

# Initialiser le modèle avec les exemples
init_model_with_examples()

# Interface utilisateur améliorée
print("\n💬 Assistant IA Français - basé sur GPT-2")
print("=" * 50)
print("📌 COMMANDES:")
print(" - Tapez votre question et appuyez sur Entrée")
print(" - Tapez 'kb' pour activer/désactiver la base de connaissances")
print(" - Tapez 'q' pour quitter")
print("=" * 50)
print(f"📚 Base de connaissances: {'ACTIVÉE' if USE_KNOWLEDGE_BASE else 'DÉSACTIVÉE'}")
print("=" * 50)

while True:
    try:
        user_input = input("\n➤ ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nAu revoir ! 👋")
            break
        
        if user_input.lower() == 'kb':
            USE_KNOWLEDGE_BASE = not USE_KNOWLEDGE_BASE
            print(f"\n📚 Base de connaissances: {'ACTIVÉE' if USE_KNOWLEDGE_BASE else 'DÉSACTIVÉE'}")
            continue
            
        if not user_input:
            continue
            
        print("\n🤔 Génération de la réponse...")
        response = generate_response(user_input)
        print("\n🤖 Réponse :")
        print(response)
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        print("Veuillez réessayer avec une autre question.")
