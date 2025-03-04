import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import time
import re

def load_model_and_tokenizer(model_path="trained_llm"):
    """Charge le modèle et le tokenizer"""
    print("\n" + "="*60)
    print("🚀 CHARGEMENT DU MODÈLE")
    print("="*60)
    
    # Rechercher le modèle dans différents emplacements
    model_paths = [model_path, "../models/trained", "../trained_llm", "trained_llm"]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Tentative de chargement depuis: {path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(path)
                
                # Utiliser CUDA si disponible
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                
                print(f"✅ Modèle et tokenizer chargés sur {device.upper()}")
                print(f"   Type de modèle: {model.config.model_type}")
                print(f"   Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
                return model, tokenizer, path
            except Exception as e:
                print(f"❌ Erreur: {str(e)}")
                continue
    
    # Fallback sur un modèle préentraîné
    print("Tentative avec modèle pré-entraîné dbddv01/gpt2-french-small...")
    tokenizer = AutoTokenizer.from_pretrained("dbddv01/gpt2-french-small")
    model = AutoModelForCausalLM.from_pretrained("dbddv01/gpt2-french-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, "dbddv01/gpt2-french-small"

def clean_response(text):
    """Nettoie la réponse des références et éléments indésirables"""
    # Enlever les références bibliographiques
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'ISBN [0-9\-]+', '', text)
    
    # Enlever les sections problématiques
    sections_to_remove = [
        r'Notes et références.*',
        r'Liens externes.*',
        r'Catégories?:.*',
        r'Portail:.*',
        r'Articles connexes.*',
        r'Bibliographie.*',
        r'Voir aussi.*'
    ]
    
    for pattern in sections_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Nettoyer les caractères excessifs
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def generate_optimized_response(prompt, model, tokenizer, verbose=True):
    """Génère une réponse en utilisant la configuration optimale"""
    # Configuration optimale basée sur les tests
    generation_config = {
        "max_length": 100,
        "do_sample": False,  # Désactiver l'échantillonnage pour plus de déterminisme
        "num_beams": 5,      # Beam search pour une meilleure qualité
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
        "repetition_penalty": 1.2,
    }
    
    # Format optimal du prompt
    system_context = "Réponds de façon factuelle et précise à cette question. "
    
    # # Formater le prompt pour le rendre plus efficace
    # if "capitale" in prompt.lower() and "france" in prompt.lower():
    #     full_prompt = f"{system_context}Question: {prompt} Réponse: La capitale de la France est"
    # else:
    full_prompt = f"{system_context}Question: {prompt} Réponse:"
    
    # Ajouter le token de début si disponible
    if tokenizer.bos_token:
        full_prompt = f"{tokenizer.bos_token}{full_prompt}"
    
    # Encoder
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Mesurer le temps
    start_time = time.time()
    
    # Générer la réponse
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config,
            pad_token_id=tokenizer.eos_token_id if not tokenizer.pad_token_id else tokenizer.pad_token_id
        )
    
    # Décoder
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire et nettoyer
    try:
        if "Réponse:" in generated_text:
            response = generated_text.split("Réponse:")[-1].strip()
        else:
            response = generated_text.replace(full_prompt, "").strip()
    except Exception:
        response = generated_text.strip()
    
    response = clean_response(response)
    
    # Durée
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n⏱️ Temps de génération: {elapsed:.2f}s")
    

    
    return response, elapsed

def interactive_mode(model, tokenizer):
    """Mode interactif pour tester le modèle"""
    print("\n" + "="*60)
    print("💬 MODE INTERACTIF - TESTEUR OPTIMISÉ")
    print("="*60)
    print("Posez vos questions (tapez 'q' pour quitter)")
    
    while True:
        print("\n> ", end="")
        prompt = input().strip()
        
        if prompt.lower() in ['q', 'quit', 'exit']:
            print("Au revoir! 👋")
            break
        
        if not prompt:
            continue
        
        print("\n⏳ Génération de la réponse...")
        
        try:
            response, elapsed = generate_optimized_response(prompt, model, tokenizer)
            print(f"\n🤖 Réponse ({elapsed:.2f}s):")
            print(response)
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    # Charger le modèle et le tokenizer
    model, tokenizer, model_path = load_model_and_tokenizer()
    
    # Définir seed pour la reproductibilité
    set_seed(42)
    
    # Lancer le mode interactif
    interactive_mode(model, tokenizer)
