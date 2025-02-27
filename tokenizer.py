import json
from transformers import AutoTokenizer
from datasets import Dataset
from config_base import MODEL_BASE, SPECIAL_TOKENS

def create_tokenizer():
    # Initialiser le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE,
        padding_side="left",
        truncation_side="left"
    )
    
    # Ajouter les tokens spéciaux
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer

# Créer et exporter le tokenizer
tokenizer = create_tokenizer()

if __name__ == "__main__":
    # Charger les données JSON
    with open("wiki_api_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Transformer en dataset compatible HuggingFace
    texts = []
    for article in data:
        if "title" in article:
            qa_text = (
                f"{tokenizer.bos_token}### Question : Qu'est-ce que {article['title']} ?\n\n"
                f"### Réponse : {article['text']}{tokenizer.eos_token}\n\n"
            )
            texts.append(qa_text)
            
            if article['title'] == "Paris":
                texts.append(
                    f"{tokenizer.bos_token}### Question : Quelle est la capitale de la France ?\n\n"
                    f"### Réponse : Paris est la capitale de la France. C'est la plus grande ville du pays "
                    f"et le centre politique, économique et culturel.{tokenizer.eos_token}\n\n"
                )

    # Tokenizer les textes avec un format plus strict
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
