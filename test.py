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
    # Préparer le prompt
    system_context = "Tu dois répondre de façon factuelle en une phrase simple. "
    full_prompt = f"{system_context}{prompt_text}"
    
    # Encoder l'entrée
    inputs = tokenizer(
        f"{tokenizer.bos_token}{full_prompt}",
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
            max_length=30,
            min_length=5,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.1,
            top_k=10,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=2,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def clean_response(text, prompt_text):
    # Nettoyer la réponse
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(prompt_text, "").strip()
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s\.,]', '', text)
    return text

# Boucle principale d'interaction
print("\nEntrez votre prompt (ou 'q' pour quitter):")
while True:
    user_input = input("> ")
    if user_input.lower() == 'q':
        print("Au revoir!")
        break
        
    if user_input.strip():
        generated = generate_response(user_input)
        cleaned = clean_response(generated, user_input)
        print("\nRéponse générée:")
        print(cleaned)
        print("\nEntrez un nouveau prompt (ou 'q' pour quitter):")
