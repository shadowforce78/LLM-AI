import torch
import torch.nn as nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model

class CustomGPT(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model(config)  # Modèle GPT-2 de base
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Tête de prédiction
        self.config = config  # Store config for easier access

    def forward(self, input_ids, attention_mask=None):
        # Get outputs from the base model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        
        # Apply the language model head
        logits = self.lm_head(hidden_states)  # Shape: [batch_size, seq_len, vocab_size]
        
        # Return logits as the main output
        return logits

# 📌 Configuration du modèle GPT-Medium (~125M params)
if __name__ == "__main__":
    config = GPT2Config(
        vocab_size=32000,  # Taille du vocabulaire (doit correspondre au tokenizer)
        n_embd=768,        # Taille des embeddings (augmentée de 512 à 768)
        n_layer=12,        # Nombre de couches (augmenté de 6 à 12)
        n_head=12,         # Nombre de têtes d'attention (augmenté de 8 à 12)
    )

    # 🔥 Instanciation du modèle
    model = CustomGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle GPT-Medium initialisé avec {total_params:,} paramètres (~{total_params/1_000_000:.1f}M).")
