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

def generate_response(prompt_text):
    # Reformater le prompt pour de meilleures réponses
    formatted_prompt = f"Question: {prompt_text}\nRéponse:"
    
    # Encoder l'entrée
    inputs = tokenizer(
        f"{tokenizer.bos_token}{formatted_prompt}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        padding=True
    )
    
    # Générer la réponse
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            min_length=10,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.3,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=2,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def clean_response(text, original_prompt):
    # Enlever les parties du prompt et le contexte
    text = text.replace("Question:", "").replace("Réponse:", "").strip()
    text = text.replace(original_prompt, "").strip()
    
    # Nettoyer la ponctuation et les caractères spéciaux
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s\.,?!]', '', text)
    
    # Enlever les segments non pertinents
    text = re.sub(r'Notes? et références.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Liens? externes?.*$', '', text, flags=re.IGNORECASE)
    
    # Nettoyer les espaces multiples et les sauts de ligne
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "Désolé, je n'ai pas de réponse claire à cette question."

# Boucle principale d'interaction
print("\nPosez votre question (ou 'q' pour quitter):")
while True:
    user_input = input("> ")
    if user_input.lower() == 'q':
        print("Au revoir!")
        break
        
    if user_input.strip():
        generated = generate_response(user_input)
        cleaned = clean_response(generated, user_input)
        print("\nRéponse:")
        print(cleaned)
        print("\nPosez votre question (ou 'q' pour quitter):")
