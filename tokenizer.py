import json
from transformers import AutoTokenizer
from datasets import Dataset

# Charger les données JSON
with open("wiki_api_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Transformer en dataset compatible HuggingFace
texts = [article["text"] for article in data]

# Configurer le tokenizer avec les tokens spéciaux
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
special_tokens_dict = {
    'pad_token': '[PAD]',
    'eos_token': '[EOS]',
    'bos_token': '[BOS]'
}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = '[PAD]'

# Tokenizer les textes avec padding et truncation
def tokenize_function(examples):
    # S'assurer que text est une liste
    texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
    
    encoded = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_special_tokens_mask=True
    )
    
    # Convertir les valeurs en listes
    return {k: [v] if not isinstance(v, list) else v for k, v in encoded.items()}

# Créer et processer le dataset
dataset = Dataset.from_dict({"text": texts})
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing texts"
)

# Sauvegarder
tokenized_datasets.save_to_disk("tokenized_dataset")
print("✅ Données tokenisées et sauvegardées !")

# Sauvegarder le tokenizer pour réutilisation
tokenizer.save_pretrained("trained_llm")
print("✅ Tokenizer sauvegardé !")
