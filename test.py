from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("trained_llm")
model = AutoModelForCausalLM.from_pretrained(
    "trained_llm",
    trust_remote_code=True,
    local_files_only=True
)
model.eval()
print("✅ Modèle et tokenizer chargés")

# Préparer l'entrée
prompt = "Quelle est la capitale de la France ?"
print(f"\nTest avec le prompt: {prompt}")

# Encoder l'entrée avec les tokens spéciaux
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    add_special_tokens=True
)

# Générer la réponse
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        min_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

# Décoder et afficher la réponse
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print("\nRéponse générée :")
print(generated_text)
