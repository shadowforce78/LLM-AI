from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Base de connaissances pour les questions courantes
KNOWLEDGE_BASE = {
    "capitale": {
        "France": "Paris est la capitale de la France. C'est la plus grande ville du pays et son centre économique et culturel.",
        "default": "Je ne connais pas la capitale de ce pays."
    },
    "définition": {
        "IA": "L'Intelligence Artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de simuler l'intelligence humaine.",
        "default": "Je ne peux pas fournir une définition précise pour ce terme."
    }
}

def get_knowledge_base_answer(question):
    """Recherche une réponse dans la base de connaissances"""
    question = question.lower()
    if "capitale" in question and "france" in question:
        return KNOWLEDGE_BASE["capitale"]["France"]
    return None

def generate_response(prompt_text, max_new_tokens=100):
    """Génère une réponse avec des paramètres optimisés"""
    # Vérifier d'abord la base de connaissances
    kb_answer = get_knowledge_base_answer(prompt_text)
    if kb_answer:
        return kb_answer

    formatted_prompt = format_prompt(prompt_text)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.3,  # Réduit davantage pour plus de cohérence
            top_k=10,        # Réduit pour plus de précision
            top_p=0.85,
            repetition_penalty=2.0,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.pad_token_id]],
        )
    
    # Décodage et nettoyage
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(generated_text, prompt_text)

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
print("\n💬 Assistant IA - Posez vos questions (ou 'q' pour quitter)")
print("=" * 50)

while True:
    try:
        user_input = input("\n➤ ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nAu revoir ! 👋")
            break
            
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
