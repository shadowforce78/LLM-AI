import torch
import os
import sys
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
import transformers  # Importer pour éviter les erreurs de type
import json

# Add project root to path to resolve imports properly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Get the project root directory for proper path resolution
def get_project_root():
    """Determine the project root directory based on execution context"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If running from src directory
    if os.path.basename(script_dir) == "src":
        return os.path.dirname(script_dir)
    
    # If running from scripts directory or elsewhere
    return os.path.abspath(os.path.join(script_dir, ".."))

# Set project root and paths
project_root = get_project_root()
dataset_path = os.path.join(project_root, "data", "tokenized_dataset")
output_dir = os.path.join(project_root, "trained_llm")
models_dir = os.path.join(project_root, "models", "trained")

# Now import modules with proper path resolution
try:
    from src.tokenizer import tokenizer
except ImportError:
    try:
        from tokenizer import tokenizer
    except ImportError:
        print("Could not import tokenizer directly, trying alternative import")
        sys.path.append(current_dir)
        from tokenizer import tokenizer

# Import model after path is configured
try:
    from models.base.modele_base import model
except ImportError:
    try:
        sys.path.append(os.path.join(project_root, "models", "base"))
        from models.base.modele_base import model
    except ImportError:
        print("Failed to import model. Please check paths and module structure.")
        exit(1)

# Vérifier les dépendances
try:
    import accelerate
    if accelerate.__version__ < "0.26.0":
        raise ImportError("accelerate>=0.26.0 required")
    
    # Vérifier si tensorboard est installé
    try:
        import tensorboard
        has_tensorboard = True
    except ImportError:
        has_tensorboard = False
        print("TensorBoard n'est pas installé. Désactivation du reporting TensorBoard.")
        print("Pour l'installer: pip install tensorboard")
except ImportError:
    raise ImportError("Please run: pip install 'accelerate>=0.26.0'")

# Charger le dataset tokenisé
print(f"Loading tokenized dataset from: {dataset_path}")
tokenized_datasets = load_from_disk(dataset_path)

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
    suggested_batch_size = max(2, min(12, int(gpu_mem/5)))  # Batch size plus petit pour plus de précision
else:
    suggested_batch_size = 2  # Batch size réduit pour plus de précision

# Définir une stratégie d'apprentissage plus précise
training_args = TrainingArguments(
    output_dir="trained_llm",
    eval_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=50,
    eval_steps=25,  # Évaluation plus fréquente
    learning_rate=5e-6,  # Learning rate plus faible pour plus de précision
    per_device_train_batch_size=suggested_batch_size,
    per_device_eval_batch_size=suggested_batch_size,
    num_train_epochs=50,  # Plus d'époques pour atteindre une convergence plus fine
    weight_decay=0.03,  # Augmentation de la régularisation
    warmup_ratio=0.15,  # Plus de warmup
    lr_scheduler_type="polynomial",  # Meilleur scheduler pour une décroissance progressive
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10,  # Logging plus fréquent
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Utiliser explicitement eval_loss comme métrique
    greater_is_better=False,
    remove_unused_columns=False,
    prediction_loss_only=True,
    gradient_accumulation_steps=4,  # Augmenter l'accumulation pour simuler des batchs plus grands
    gradient_checkpointing=True,
    max_grad_norm=1.0,  # Gradient clipping pour stabiliser l'entraînement
    group_by_length=True,  # Regrouper les séquences de longueurs similaires
    report_to=["tensorboard"] if has_tensorboard else [],  # Activer TensorBoard uniquement s'il est installé
)

# Ajuster le data collator pour le batch size
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# Configurer les callbacks en fonction des dépendances disponibles
callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=5)]  # Early stopping toujours activé

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
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
print(f"├─ Weight decay: {training_args.weight_decay}")
print(f"├─ Max gradient norm: {training_args.max_grad_norm}")
print(f"└─ LR scheduler: {training_args.lr_scheduler_type}")

# Modification du try-except pour plus de clarté
try:
    print("\n🏃 DÉMARRAGE DE L'ENTRAÎNEMENT...")
    print(f"└─ Paramètres du modèle: {format_number(sum(p.numel() for p in model.parameters()))}")
    
    # Créer une classe de callback personnalisée pour le logging
    class CustomCallback(transformers.TrainerCallback):
        def __init__(self):
            self.best_loss = float('inf')
            
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
                    
                    # Tracking best loss
                    if isinstance(eval_loss, (int, float)) and eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        improvement = "🔻 (nouveau meilleur!)"
                    else:
                        improvement = ""
                    
                    print(f"├─ Loss: {loss_str}")
                    print(f"└─ Eval Loss: {eval_loss_str} {improvement}")

    # Ajouter le callback au trainer
    trainer.add_callback(CustomCallback())
    
    # Lancer l'entraînement
    trainer.train()
    
    print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*50)
    
    # Sauvegarder avec les métadonnées nécessaires
    output_dir = "models/trained"
    os.makedirs(output_dir, exist_ok=True)
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
