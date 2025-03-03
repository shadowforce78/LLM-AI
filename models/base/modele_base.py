import torch
import os
import sys
from transformers import AutoModelForCausalLM

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))  # Add project root to path

# Try different import approaches to ensure it works regardless of execution context
try:
    from models.base.config_base import MODEL_BASE
except ImportError:
    try:
        from config_base import MODEL_BASE
    except ImportError:
        # Last resort, direct path import
        sys.path.append(current_dir)
        from config_base import MODEL_BASE

# Try to import tokenizer with flexible path handling
try:
    from src.tokenizer import tokenizer
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(current_dir)), "src"))
        from src.tokenizer import tokenizer
    except ImportError:
        print("Warning: Could not import tokenizer. Using fallback approach.")
        # Create a minimal tokenizer as fallback
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

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
