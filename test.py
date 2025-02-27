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
    # Ajout de contexte et de structure
    formatted = f"{tokenizer.bos_token}Question : {text}\nRéponse : "
    return formatted

def generate_response(prompt_text, max_new_tokens=100):
    """Génère une réponse avec des paramètres optimisés"""
    formatted_prompt = format_prompt(prompt_text)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Décodage et nettoyage
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(generated_text, prompt_text)

def clean_response(text, original_prompt):
    """Nettoie et formate la réponse générée"""
    # Extraire la réponse
    try:
        response = text.split("Réponse :")[1].strip()
    except IndexError:
        response = text.replace(original_prompt, "").strip()
    
    # Nettoyage basique
    response = re.sub(r'\s+', ' ', response)
    response = re.sub(r'^\W+', '', response)
    response = re.sub(r'\W+$', '', response)
    
    # Vérifier la qualité de la réponse
    if len(response) < 10 or response.count(' ') < 2:
        return "Je ne peux pas générer une réponse cohérente à cette question."
    
    return response

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
