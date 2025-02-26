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

# Améliorer le format du prompt avec un contexte
system_context = "Tu es un assistant IA qui répond de façon claire et concise aux questions. "
prompt = "Question: Quelle est la capitale de la France ? Réponse: La capitale de la France est"
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

# Ajuster les paramètres de génération
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,  # Encore plus court pour forcer la concision
        min_length=10,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.3,  # Encore plus bas pour plus de déterminisme
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.4,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=3,
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
    
    # Enlever les références et liens
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'Liens? externes?.*$', '', text, flags=re.MULTILINE|re.DOTALL)
    text = re.sub(r'[^:]*: (?=La capitale)', '', text)
    
    # Enlever les citations et références bibliographiques
    text = re.sub(r'« [^»]* »', '', text)
    text = re.sub(r'lire en ligne.*$', '', text, flags=re.MULTILINE)
    
    # Nettoyer le formatage
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s*(?=[A-Z])', r'\1\n', text)
    
    # Garder uniquement la première phrase pertinente
    sentences = text.split('.')
    relevant_sentence = next((s for s in sentences if 'Paris' in s or 'capitale' in s.lower()), sentences[0])
    
    return relevant_sentence.strip() + "."

cleaned_text = clean_response(generated_text)
print("\nRéponse générée :")
print(cleaned_text)
