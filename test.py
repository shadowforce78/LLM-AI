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

# Améliorer le format du prompt
system_context = "Tu dois répondre de façon factuelle en une phrase simple. "
prompt = "Complète la phrase suivante: La capitale de la France est"
full_prompt = f"{system_context}{prompt}"
print(f"\nTest avec le prompt: {prompt}")

# Encoder l'entrée avec les tokens spéciaux
inputs = tokenizer(
    f"{tokenizer.bos_token}{full_prompt}",
    return_tensors="pt",
    truncation=True,
    max_length=512,
    add_special_tokens=True,
    padding=True
)

# Configurer les paramètres de génération de manière cohérente
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=30,
        min_length=5,
        num_return_sequences=1,
        do_sample=True,  # Activé pour utiliser temperature et top_k
        temperature=0.1,
        top_k=10,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=2,  # Augmenté pour utiliser early_stopping
        early_stopping=True
    )

# Décoder et afficher la réponse
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

# Post-traitement pour nettoyer la sortie
def clean_response(text):
    # Enlever le contexte et le prompt
    text = text.replace(system_context, "").replace(prompt, "").strip()
    
    # Nettoyer la réponse
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'.*?[Ll]a capitale de la France est\s*', '', text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s\.]', '', text)
    
    # Retourner la réponse nettoyée
    return text

cleaned_text = clean_response(generated_text)
print("\nRéponse générée :")
print(cleaned_text)
