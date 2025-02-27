import json
import os
import sys
from transformers import AutoTokenizer
from datasets import Dataset

# Adjust imports based on execution context
try:
    from models.base.config_base import MODEL_BASE, SPECIAL_TOKENS
except ImportError:
    # Add parent directory to path when running from scripts folder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.base.config_base import MODEL_BASE, SPECIAL_TOKENS


def create_tokenizer():
    # Initialiser le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE, padding_side="left", truncation_side="left"
    )

    # Ajouter les tokens spéciaux
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


# Créer et exporter le tokenizer
tokenizer = create_tokenizer()


def get_project_root():
    """Determine the project root directory based on execution context"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If running from src directory
    if os.path.basename(script_dir) == "src":
        return os.path.dirname(script_dir)

    # If running from scripts directory or elsewhere
    return os.path.abspath(os.path.join(script_dir, ".."))


if __name__ == "__main__":
    # Get project root directory
    project_root = get_project_root()
    print(f"Project root identified as: {project_root}")

    # Define paths relative to project root
    raw_data_path = os.path.join(project_root, "data", "raw", "wiki_dataset.json")
    tokenized_data_path = os.path.join(project_root, "data", "tokenized_dataset")
    model_path = os.path.join(project_root, "models", "trained", "trained_llm")

    # Ensure directories exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(tokenized_data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    print(f"Loading data from: {raw_data_path}")
    # Charger les données JSON
    try:
        with open(raw_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} articles from dataset")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {raw_data_path}")
        print("Please run the wiki-scrap.py script first to generate the dataset.")
        sys.exit(1)

    # Transformer en dataset compatible HuggingFace
    texts = []
    for article in data:
        if "title" in article:
            qa_text = (
                f"{tokenizer.bos_token}### Question : Qu'est-ce que {article['title']} ?\n\n"
                f"### Réponse : {article['text']}{tokenizer.eos_token}\n\n"
            )
            texts.append(qa_text)

            if article["title"] == "Paris":
                texts.append(
                    f"{tokenizer.bos_token}### Question : Quelle est la capitale de la France ?\n\n"
                    f"### Réponse : Paris est la capitale de la France. C'est la plus grande ville du pays "
                    f"et le centre politique, économique et culturel.{tokenizer.eos_token}\n\n"
                )

    print(f"Created {len(texts)} text samples for tokenization")

    # Tokenizer les textes avec un format plus strict
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )

    # Créer et processer le dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing texts",
    )

    # Sauvegarder
    print(f"Saving tokenized dataset to: {tokenized_data_path}")
    tokenized_datasets.save_to_disk(tokenized_data_path)
    print("✅ Données tokenisées et sauvegardées !")

    print(f"Saving tokenizer to: {model_path}")
    tokenizer.save_pretrained(model_path)
    print("✅ Tokenizer sauvegardé !")
