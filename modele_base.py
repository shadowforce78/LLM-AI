import torch
from transformers import AutoModelForCausalLM, GPT2Config
from tokenizer import tokenizer

# Charger le modèle pré-entraîné français
model = AutoModelForCausalLM.from_pretrained(
    "dbddv01/gpt2-french-small",
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Redimensionner l'embedding pour correspondre à notre tokenizer
model.resize_token_embeddings(len(tokenizer))

print("✅ Modèle GPT-2 français chargé !")
