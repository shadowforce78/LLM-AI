import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
from tokenizer import tokenizer
from modele_base import *
import json

# Vérifier les dépendances
try:
    import accelerate

    if accelerate.__version__ < "0.26.0":
        raise ImportError("accelerate>=0.26.0 required")
except ImportError:
    raise ImportError("Please run: pip install 'accelerate>=0.26.0'")

# Charger le dataset tokenisé
tokenized_datasets = load_from_disk("tokenized_dataset")

# Ajuster le split en fonction de la taille du dataset
n_samples = len(tokenized_datasets)
if n_samples < 10:
    # Pour les petits datasets, garder un seul exemple pour le test
    split_datasets = tokenized_datasets.train_test_split(test_size=1)
else:
    # Split normal pour les datasets plus grands
    split_datasets = tokenized_datasets.train_test_split(test_size=0.1)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

print(f"Taille du dataset total: {n_samples}")
print(f"Taille du dataset d'entraînement: {len(train_dataset)}")
print(f"Taille du dataset de test: {len(eval_dataset)}")

# Vérifier que CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

training_args = TrainingArguments(
    output_dir="trained_llm",
    eval_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=100,
    eval_steps=100,
    learning_rate=5e-5,  # Taux d'apprentissage plus faible pour le fine-tuning
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Plus d'epochs
    weight_decay=0.01,
    warmup_steps=500,  # Ajout d'un warmup
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    remove_unused_columns=False,
    prediction_loss_only=True,
)

# Data collator ajusté
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Ajout d'un try-except plus détaillé
try:
    print("Starting training...")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Vérifier le format des données
    print("Sample input:", next(iter(train_dataset)))

    trainer.train()

    # Sauvegarder avec les métadonnées nécessaires
    output_dir = "trained_llm"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    # Sauvegarder les métriques d'entraînement
    training_stats = {
        "final_loss": trainer.state.log_history[-1].get("loss"),
        "best_eval_loss": trainer.state.best_metric,
        "num_epochs": trainer.state.num_train_epochs,
        "total_steps": trainer.state.global_step,
        "model_type": "gpt2",
        "base_model": "dbddv01/gpt2-french-small",
    }

    with open(f"{output_dir}/training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=2)

    print("✅ Modèle entraîné et sauvegardé !")

except Exception as e:
    print(f"❌ Erreur pendant l'entraînement : {str(e)}")
    import traceback

    traceback.print_exc()
