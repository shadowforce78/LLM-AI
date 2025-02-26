import json
from transformers import AutoTokenizer
from datasets import Dataset

# Charger les données JSON
with open("wiki_api_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Transformer en dataset compatible HuggingFace
texts = [article["text"] for article in data]

# Utiliser le tokenizer GPT-2 français
tokenizer = AutoTokenizer.from_pretrained(
    "dbddv01/gpt2-french-small",
    padding_side="left",
    truncation_side="left"
)

# Configurer les tokens spéciaux
special_tokens = {
    "pad_token": "<|pad|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
}
tokenizer.add_special_tokens(special_tokens)

# Tokenizer les textes
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True,
        add_special_tokens=True
    )

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
tokenizer.save_pretrained("trained_llm")
print("✅ Tokenizer sauvegardé !")
