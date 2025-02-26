import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
from tokenizer import tokenizer
from modele_base import *
import transformers # Importer pour éviter les erreurs de type
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

# Vérifier la mémoire GPU disponible pour ajuster le batch size
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # en GB
    suggested_batch_size = max(4, min(16, int(gpu_mem/4)))  # 4GB par batch environ
else:
    suggested_batch_size = 4

training_args = TrainingArguments(
    output_dir="trained_llm",
    eval_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=50,  # Sauvegarde plus fréquente
    eval_steps=50,
    learning_rate=1e-5,  # Learning rate réduit pour plus de stabilité
    per_device_train_batch_size=suggested_batch_size,
    per_device_eval_batch_size=suggested_batch_size,
    num_train_epochs=15,  # Plus d'epochs pour compenser le learning rate plus faible
    weight_decay=0.02,  # Augmenté pour une meilleure régularisation
    warmup_ratio=0.1,  # Utilisation d'un ratio plutôt qu'un nombre fixe de steps
    lr_scheduler_type="cosine",  # Scheduler plus progressif
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    remove_unused_columns=False,
    prediction_loss_only=True,
    gradient_accumulation_steps=2,  # Accumulation des gradients pour simuler un batch size plus grand
    gradient_checkpointing=True,  # Économie de mémoire
)

# Ajuster le data collator pour le batch size
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Ajouter une fonction pour formater les nombres
def format_number(num):
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

print("\n" + "="*50)
print("🚀 INITIALISATION DE L'ENTRAÎNEMENT")
print("="*50)

# Affichage des informations sur le dataset
print("\n📊 INFORMATIONS DU DATASET:")
print(f"├─ Taille totale: {format_number(n_samples)} exemples")
print(f"├─ Dataset d'entraînement: {format_number(len(train_dataset))} exemples")
print(f"└─ Dataset de test: {format_number(len(eval_dataset))} exemples")

# Affichage des informations sur le matériel
print("\n💻 CONFIGURATION MATÉRIELLE:")
print(f"├─ Device: {device.upper()}")
if torch.cuda.is_available():
    print(f"├─ GPU: {torch.cuda.get_device_name(0)}")
    print(f"└─ Mémoire GPU: {format_number(gpu_mem)}GB")
else:
    print("└─ Mode CPU uniquement")

# Affichage des paramètres d'entraînement
print("\n⚙️ PARAMÈTRES D'ENTRAÎNEMENT:")
print(f"├─ Batch size effectif: {suggested_batch_size * training_args.gradient_accumulation_steps}")
print(f"├─ Learning rate: {training_args.learning_rate:.2e}")
print(f"├─ Nombre d'epochs: {int(training_args.num_train_epochs)}")
print(f"├─ Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"└─ Weight decay: {training_args.weight_decay}")

# Modification du try-except pour plus de clarté
try:
    print("\n🏃 DÉMARRAGE DE L'ENTRAÎNEMENT...")
    print(f"└─ Paramètres du modèle: {format_number(sum(p.numel() for p in model.parameters()))}")
    
    # Créer une classe de callback personnalisée pour le logging
    class CustomCallback(transformers.TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch.is_integer():
                print(f"\n📈 Epoch {int(state.epoch)}/{int(training_args.num_train_epochs)}:")
                if state.log_history:
                    latest_logs = state.log_history[-1]
                    loss = latest_logs.get('loss')
                    eval_loss = latest_logs.get('eval_loss')
                    
                    # Format losses only if they are numbers
                    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
                    eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else "N/A"
                    
                    print(f"├─ Loss: {loss_str}")
                    print(f"└─ Eval Loss: {eval_loss_str}")

    # Ajouter le callback au trainer
    trainer.add_callback(CustomCallback())
    
    # Lancer l'entraînement
    trainer.train()
    
    print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*50)
    
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
    print("\n❌ ERREUR PENDANT L'ENTRAÎNEMENT")
    print("="*50)
    print(f"Nature de l'erreur: {str(e)}")
    import traceback

    traceback.print_exc()
