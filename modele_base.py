import torch
from transformers import AutoModelForCausalLM
from config_base import MODEL_BASE
from tokenizer import tokenizer

def create_model():
    # Charger le modèle pré-entraîné français
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Redimensionner l'embedding
    model.resize_token_embeddings(len(tokenizer))
    return model

# Créer et exporter le modèle
model = create_model()
print("✅ Modèle GPT-2 français chargé !")
