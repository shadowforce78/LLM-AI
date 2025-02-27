import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path="trained_llm"):
    """Charge le modèle et le tokenizer"""
    print("⏳ Chargement du modèle et du tokenizer...")
    
    # Charger le tokenizer et le modèle depuis les fichiers sauvegardés
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Mode d'évaluation
    
    return model, tokenizer, device

def generate_response(model, tokenizer, device, question, max_length=150):
    """Génère une réponse basée uniquement sur le modèle entraîné"""
    # Formatage du prompt
    prompt = f"{tokenizer.bos_token}### Question : {question}\n\n### Réponse :"
    
    # Tokenisation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True
    ).to(device)
    
    # Génération
    print("🧠 Réflexion en cours...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs.input_ids.size(1) + max_length,
            do_sample=True,
            temperature=0.8,  # Température légèrement plus élevée pour plus de créativité
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            num_beams=3,  # Beam search pour une meilleure cohérence
            early_stopping=True,
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
    
    # Nettoyage de base
    response = clean_response(response)
    
    return response

def clean_response(text):
    """Nettoie la réponse sans supprimer trop d'informations"""
    import re
    
    # Supprimer les marqueurs et autres artefacts
    text = re.sub(r'###.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-•*]\s*', '', text, flags=re.MULTILINE)  # Listes à puces
    
    # Supprimer les notes et références typiques de Wikipedia
    patterns_to_remove = [
        r'Notes et références.*$',
        r'Liens externes.*$',
        r'Voir aussi.*$',
        r'Bibliographie.*$',
        r'Articles connexes.*$',
        r'^\s*\[\d+\]',  # Citations numériques
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL|re.MULTILINE)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Interface utilisateur
if __name__ == "__main__":
    # Charger le modèle et le tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"✅ Modèle chargé sur {device.upper()}")
    
    print("\n💬 Assistant IA - Génération Pure")
    print("=" * 50)
    print("🧠 Cet assistant répond uniquement en utilisant ses connaissances apprises")
    print("💡 Il n'utilise aucune base de connaissances prédéfinie")
    print("❓ Tapez 'q' pour quitter")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n➤ ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nAu revoir! 👋")
                break
                
            if not user_input:
                continue
            
            response = generate_response(model, tokenizer, device, user_input)
            print("\n🤖 Réponse générée:")
            print(response)
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"\n❌ Erreur: {str(e)}")
            print("Veuillez réessayer avec une autre question.")
