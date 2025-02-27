from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Configuration
MODEL_PATH = "models/trained"  # Chemin principal du modèle entraîné
FALLBACK_PATH = "trained_llm"  # Chemin alternatif

def init_model_and_tokenizer():
    """Initialise le modèle et le tokenizer"""
    print("⏳ Chargement du modèle et du tokenizer...")
    
    try:
        # Vérifier le chemin principal d'abord
        import os
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_PATH
        print(f"Utilisation du modèle depuis: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configuration des tokens spéciaux
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé sur {device.upper()}")
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None, "cpu"

def format_prompt(text):
    """Formatte le prompt pour une meilleure génération"""
    return f"### Question : {text}\n\n### Réponse : "

def generate_model_response(model, tokenizer, device, question, max_length=150):
    """Génère une réponse basée uniquement sur le modèle entraîné"""
    # Formatage du prompt
    prompt = format_prompt(question)
    
    # Tokenisation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True
    ).to(device)
    
    # Génération
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs.input_ids.size(1) + max_length,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                num_beams=1,  # Simple greedy decoding pour la vitesse
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Décodage de la sortie
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraction de la réponse
        if "### Réponse :" in generated_text:
            response = generated_text.split("### Réponse :")[1].strip()
        else:
            response = generated_text.split(question)[1].strip()
        
        # Nettoyage
        response = clean_response(response)
        
        return response
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {str(e)}")
        return "Je ne peux pas répondre à cette question pour le moment."

def clean_response(text):
    """Nettoie la réponse générée"""
    # Suppression des artefacts communs
    patterns_to_remove = [
        r'###.*?$',
        r'Notes et références.*$',
        r'Liens externes.*$',
        r'Voir aussi.*$',
        r'Bibliographie.*$',
        r'Articles connexes.*$',
        r'^\s*\[\d+\]'
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL|re.MULTILINE)
    
    # Normaliser les espaces et nettoyer
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Vérification de la qualité minimale
    if len(text) < 5 or text.count(' ') < 1:
        return "Je n'ai pas de réponse précise à cette question."
        
    return text

# Interface utilisateur
def main():
    # Initialisation
    model, tokenizer, device = init_model_and_tokenizer()
    
    print("\n💬 Assistant IA Français")
    print("=" * 50)
    print("📌 COMMANDES:")
    print(" - Tapez votre question et appuyez sur Entrée")
    print(" - Tapez 'q' pour quitter")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n➤ ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nAu revoir! 👋")
                break
            
            # Traitement avec le modèle
            if model is not None:
                print("\n🧠 Réflexion en cours...")
                response = generate_model_response(model, tokenizer, device, user_input)
                print("\n🤖 Réponse (générée par le modèle):")
                print(response)
            else:
                print("\n❌ Modèle non disponible.")
                
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            print("\nOpération annulée par l'utilisateur.")
            print("\nAu revoir! 👋")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {str(e)}")
            print("Veuillez réessayer.")

if __name__ == "__main__":
    main()
