import torch
import os
import sys
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
import transformers  # Importer pour éviter les erreurs de type
import json
import datetime  # Pour générer des noms de dossier uniques basés sur la date/heure
import uuid  # Pour générer des identifiants uniques
import warnings
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import multiprocessing  # Importer explicitement multiprocessing

# Supprimer les avertissements spécifiques qui se répètent
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

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
output_dir = os.path.join(project_root, "models", "trained", "trained_llm")
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

# Générer un nom unique pour les logs TensorBoard avec plus de détails
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
unique_id = str(uuid.uuid4())[:8]  # Prendre les 8 premiers caractères de l'UUID
# Inclure des informations sur le dataset et le modèle dans le nom du dossier
model_name = "sushiAI"
dataset_info = os.path.basename(os.path.dirname(dataset_path))
tensorboard_log_dir = f"./logs/{model_name}_{dataset_info}_{current_time}_{unique_id}"


# Fonction d'optimisation du scheduler pour une décroissance plus contrôlée
def get_optimized_scheduler(optimizer, warmup_steps, total_steps):
    """
    Crée un scheduler à décroissance linéaire puis cosinus pour réduire la perte plus efficacement
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Phase d'échauffement: croissance linéaire de 0 à 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Phase de décroissance cosinus: décroissance douce et progressive
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# Fonction pour vérifier si un dataset nécessite plus d'attention
def analyze_dataset_complexity(dataset):
    """
    Analyse la complexité du dataset pour ajuster les hyperparamètres en conséquence
    """
    # Vérifier la longueur moyenne des textes
    if "input_ids" in dataset[0]:
        seq_lengths = [len(example["input_ids"]) for example in dataset]
        avg_seq_length = sum(seq_lengths) / len(seq_lengths)
        max_seq_length = max(seq_lengths)

        complexity = {
            "avg_seq_length": avg_seq_length,
            "max_seq_length": max_seq_length,
            "is_complex": avg_seq_length > 256 or max_seq_length > 512,
        }

        return complexity

    return {"is_complex": False}


# Charger le dataset tokenisé
print(f"Loading tokenized dataset from: {dataset_path}")
tokenized_datasets = load_from_disk(dataset_path)
dataset_complexity = analyze_dataset_complexity(tokenized_datasets)

# Ajuster le split en fonction de la taille du dataset
n_samples = len(tokenized_datasets)

# Pour les datasets plus grands, utiliser une validation plus importante
if n_samples > 1000:
    val_size = 0.15  # Plus de validation pour capter la généralisation
elif n_samples < 10:
    val_size = 1  # Pour les très petits datasets
else:
    val_size = 0.1

# Split avec stratification si possible
try:
    split_datasets = tokenized_datasets.train_test_split(
        test_size=val_size, shuffle=True, seed=42
    )
except:
    split_datasets = tokenized_datasets.train_test_split(test_size=val_size)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

print(f"Taille du dataset total: {n_samples}")
print(f"Taille du dataset d'entraînement: {len(train_dataset)}")
print(f"Taille du dataset de test: {len(eval_dataset)}")

# Vérifier que CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optimiser le batch size en fonction de la longueur des séquences
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Ajuster le batch size en fonction de la complexité du dataset
    if dataset_complexity["is_complex"]:
        suggested_batch_size = max(
            1, min(6, int(gpu_mem / 8))
        )  # Plus petit pour séquences longues
    else:
        suggested_batch_size = max(
            2, min(16, int(gpu_mem / 4))
        )  # Plus grand pour séquences courtes
else:
    suggested_batch_size = 2

# Déterminer le learning rate optimal en fonction de la taille du dataset
if n_samples < 100:
    base_lr = 8e-5  # Learning rate plus élevé pour petits datasets
elif n_samples < 1000:
    base_lr = 5e-5
else:
    base_lr = 2e-5  # Learning rate plus faible pour grands datasets

# Calculer le nombre d'étapes et d'époques optimal
epochs = min(
    100, max(20, int(10000 / max(1, n_samples)))
)  # Plus d'époques pour petits datasets
steps_per_epoch = math.ceil(
    len(train_dataset) / (suggested_batch_size * 4)
)  # 4 = gradient_accumulation_steps
total_training_steps = steps_per_epoch * epochs

# Assurer que les steps soient compatibles pour load_best_model_at_end
eval_steps = max(10, steps_per_epoch // 4)  # Évaluation fréquente
# Assurer que save_steps est un multiple entier de eval_steps
save_steps = (
    math.ceil(steps_per_epoch / math.ceil(steps_per_epoch / eval_steps)) * eval_steps
)

# Afficher les étapes pour faciliter le débogage
print(f"Steps par epoch: {steps_per_epoch}")
print(f"Eval steps: {eval_steps}")
print(f"Save steps: {save_steps} (multiple de {eval_steps})")

# Configuration optimisée de TensorBoard
tensorboard_config = {
    "enabled": has_tensorboard,
    "log_dir": tensorboard_log_dir if has_tensorboard else "./logs",
    "flush_secs": 20,
    "update_freq": "epoch",
}


# Configurer multiprocessing correctement
# Cette fonction aide à éviter les problèmes avec Windows
def setup_multiprocessing():
    """Configure multiprocessing selon la plateforme"""
    if sys.platform == "win32":
        # Sur Windows, utiliser 'spawn' au lieu de 'fork'
        multiprocessing.set_start_method("spawn", force=True)
        return False  # Désactiver les workers sur Windows par défaut
    else:
        return True  # Activer les workers sur Unix/Linux


# Déterminer si on peut utiliser des workers en toute sécurité
can_use_workers = setup_multiprocessing()

# Définir une stratégie d'apprentissage hautement optimisée pour réduire la perte
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=save_steps,  # Utiliser la valeur calculée compatible
    eval_steps=eval_steps,  # Utiliser la valeur calculée
    learning_rate=base_lr,
    per_device_train_batch_size=suggested_batch_size,
    per_device_eval_batch_size=suggested_batch_size,
    num_train_epochs=epochs,
    weight_decay=0.05,  # Régularisation plus forte pour meilleure généralisation
    warmup_ratio=(
        0.1 if n_samples > 500 else 0.2
    ),  # Plus long warmup pour petits datasets
    lr_scheduler_type="cosine_with_restarts",  # Scheduler avec restarts pour sortir des minima locaux
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    logging_dir=tensorboard_config["log_dir"],
    logging_steps=eval_steps,  # Utiliser la même valeur que eval_steps pour la cohérence
    load_best_model_at_end=True,
    metric_for_best_model="loss",  # Utiliser la perte d'entraînement comme métrique
    greater_is_better=False,
    remove_unused_columns=False,
    prediction_loss_only=True,
    gradient_accumulation_steps=4,  # Simuler des batchs plus grands
    gradient_checkpointing=(
        True if dataset_complexity["is_complex"] else False
    ),  # Selon complexité
    max_grad_norm=1.0,  # Clipping des gradients
    group_by_length=True,  # Optimisation de performance
    report_to=["tensorboard"] if tensorboard_config["enabled"] else [],
    # Ne pas utiliser de workers sur Windows ou si can_use_workers est False
    dataloader_num_workers=0 if sys.platform == "win32" or not can_use_workers else 4,
    ddp_find_unused_parameters=False,  # Optimisation pour les modèles parallèles
    optim="adamw_torch",  # Utiliser l'implémentation PyTorch d'AdamW
    bf16=False,  # Bfloat16 si disponible, mais pas pour la compatibilité
)

# Ajuster le data collator pour une tokenisation plus efficace
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)

# Configurer les callbacks avec analyse de performance améliorée
callbacks = []

# Early stopping plus patient pour éviter l'arrêt prématuré
patience = (
    25 if n_samples > 1000 else 15
)  # Augmenté considérablement pour éviter l'arrêt prématuré
callbacks.append(
    transformers.EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0005,  # Seuil plus bas pour éviter l'arrêt trop rapide
    )
)


# Callback personnalisé pour le logging et sauvegarde des checkpoints avancés
class OptimizedTrainingCallback(transformers.TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")
        self.best_eval_loss = float("inf")
        self.epoch_losses = []
        self.plateau_count = 0
        self.last_improvement = 0
        self.force_continue = (
            True  # Forcer la continuation pendant un certain nombre d'epochs
        )
        self.min_epochs = min(
            50, int(training_args.num_train_epochs * 0.5)
        )  # Au moins la moitié des époques totales

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.epoch.is_integer():
            return

        epoch = int(state.epoch)
        print(f"\n📈 Epoch {epoch}/{int(training_args.num_train_epochs)}:")

        # Forcer l'entraînement à continuer pendant min_epochs, même si early stopping serait déclenché
        if epoch < self.min_epochs:
            control.should_training_stop = False
            print(
                f"ℹ️ Minimum d'epochs pas encore atteint ({epoch}/{self.min_epochs}), poursuite forcée de l'entraînement"
            )

        if state.log_history:
            # Extraire les métriques de performance
            latest_logs = state.log_history[-1]
            loss = latest_logs.get("loss")
            eval_loss = latest_logs.get("eval_loss")

            # Format des pertes pour affichage
            loss_str = f"{loss:.6f}" if isinstance(loss, (int, float)) else "N/A"
            eval_loss_str = (
                f"{eval_loss:.6f}" if isinstance(eval_loss, (int, float)) else "N/A"
            )

            # Suivi de la performance et des améliorations
            if isinstance(eval_loss, (int, float)):
                self.epoch_losses.append(eval_loss)
                improved = False

                # Vérifier si c'est la meilleure perte d'évaluation
                if eval_loss < self.best_eval_loss:
                    absolute_improvement = self.best_eval_loss - eval_loss
                    relative_improvement = (
                        absolute_improvement / self.best_eval_loss
                        if self.best_eval_loss > 0
                        else 0
                    )

                    self.best_eval_loss = eval_loss
                    improvement = f"🔻 (-{absolute_improvement:.6f}, {relative_improvement*100:.2f}%, nouveau record!)"
                    self.last_improvement = epoch
                    improved = True
                    self.plateau_count = 0

                    # Sauvegarde du meilleur modèle avec score précis
                    best_model_dir = os.path.join(
                        models_dir, f"best_model_{eval_loss:.6f}"
                    )
                    if not os.path.exists(best_model_dir):
                        os.makedirs(best_model_dir, exist_ok=True)
                        trainer.save_model(best_model_dir)
                        print(f"✅ Meilleur modèle sauvegardé: {best_model_dir}")
                else:
                    plateau_length = epoch - self.last_improvement
                    self.plateau_count += 1

                    # Modifier le message selon la durée du plateau et l'avancement global
                    if plateau_length > patience // 2:
                        improvement = f"⚠️ (plateau: {plateau_length}/{patience} epochs)"
                    else:
                        improvement = f"(plateau: {plateau_length} epochs)"

                    # Suggestions pour sortir d'un plateau
                    if self.plateau_count >= 3 and self.plateau_count % 3 == 0:
                        print(
                            f"💡 Suggestion: plateau détecté depuis {plateau_length} epochs."
                        )
                        if trainer.optimizer.param_groups[0]["lr"] > 1e-6:
                            progress_percent = (
                                epoch / training_args.num_train_epochs * 100
                            )

                            if progress_percent < 50:
                                print(
                                    "   Considérez attendre, l'entraînement est encore dans sa première moitié."
                                )
                            elif plateau_length > patience // 2:
                                print(
                                    "   Attention: l'early stopping pourrait se déclencher bientôt."
                                )
            else:
                improvement = ""

            # Afficher statistiques d'entraînement détaillées
            print(f"├─ Train Loss: {loss_str}")
            print(f"├─ Eval Loss: {eval_loss_str} {improvement}")

            # Afficher le learning rate actuel
            if hasattr(trainer, "optimizer") and trainer.optimizer:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                print(f"├─ Learning rate: {current_lr:.2e}")

            # Graphique d'évolution de la perte
            if len(self.epoch_losses) > 1:
                print(f"📊 Évolution de la perte: ", end="")
                self._print_loss_trend()

    def on_step_end(self, args, state, control, **kwargs):
        """Interception des étapes pour éviter un arrêt prématuré forcé"""
        # S'assurer que l'entraînement ne s'arrête pas avant le nombre minimum d'époques
        current_epoch = state.epoch
        if current_epoch < self.min_epochs and control.should_training_stop:
            print(
                f"⚠️ Tentative d'arrêt prématuré à l'epoch {current_epoch:.2f} bloquée."
            )
            control.should_training_stop = False

    def _print_loss_trend(self):
        # Graphique ASCII amélioré avec marqueurs de tendance
        if len(self.epoch_losses) < 2:
            return

        # Utiliser les 30 derniers points au maximum pour l'affichage
        losses = self.epoch_losses[-30:]
        min_loss = min(losses)
        max_loss = max(losses)

        # Éviter la division par zéro
        range_loss = max_loss - min_loss
        if range_loss < 0.0001:  # Quasi-plateau
            range_loss = 0.0001

        # Calculer la tendance (diminution, stagnation ou augmentation)
        if len(losses) >= 5:
            recent_avg = sum(losses[-5:]) / 5
            earlier_avg = sum(losses[-10:-5]) / 5 if len(losses) >= 10 else losses[0]
            trend = recent_avg - earlier_avg

            if trend < -0.001:
                trend_indicator = " ↘️ (amélioration)"
            elif trend > 0.001:
                trend_indicator = " ↗️ (dégradation)"
            else:
                trend_indicator = " → (stable)"
        else:
            trend_indicator = ""

        # Caractères pour le graphique
        chars = "▁▂▃▄▅▆▇█"
        for loss in losses:
            # Normaliser et inverser (plus bas = meilleur)
            normalized = 1.0 - ((loss - min_loss) / range_loss)
            idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            print(chars[idx], end="")

        print(trend_indicator)


# Ajouter le callback personnalisé
callbacks.append(OptimizedTrainingCallback())

# Configurer le trainer avec les callbacks optimisés
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
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"


print("\n" + "=" * 50)
print("🚀 INITIALISATION DE L'ENTRAÎNEMENT")
print("=" * 50)

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

# Affichage des paramètres d'entraînement optimisés
print("\n⚙️ CONFIGURATION D'ENTRAÎNEMENT OPTIMISÉE:")
print(
    f"├─ Batch size effectif: {suggested_batch_size * training_args.gradient_accumulation_steps}"
)
print(f"├─ Learning rate: {training_args.learning_rate:.2e}")
print(f"├─ Nombre d'epochs: {int(training_args.num_train_epochs)}")
print(f"├─ Evaluation steps: {eval_steps}")
print(f"├─ Save steps: {save_steps} (multiple de {eval_steps})")
print(f"├─ Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"├─ Weight decay: {training_args.weight_decay}")
print(f"├─ Scheduler: {training_args.lr_scheduler_type}")
print(f"├─ Warmup ratio: {training_args.warmup_ratio}")
print(f"└─ Steps total: ~{total_training_steps}")

# Modification du try-except pour plus de clarté
try:
    # Définir une fonction d'initialisation de l'optimiseur personnalisée (optionnel)
    def custom_optimizer_init(model):
        """Crée un optimiseur spécialement configuré pour minimiser la perte"""
        # Séparer les paramètres du modèle en groupes avec decay et sans decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Créer l'optimiseur avec des paramètres plus agressifs
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            betas=(0.9, 0.99),  # Beta2 plus élevé pour une meilleure stabilité
            eps=1e-8,
        )
        return optimizer

    # Optionnel: définir l'optimiseur personnalisé
    # trainer.create_optimizer = custom_optimizer_init

    # Lancer l'entraînement avec protections pour multiprocessing
    print("\n🚀 LANCEMENT DE L'ENTRAÎNEMENT OPTIMISÉ...")
    print(
        f"├─ Complexité du dataset: {'Élevée' if dataset_complexity['is_complex'] else 'Standard'}"
    )
    print(
        f"└─ Workers: {'Désactivés (Windows)' if sys.platform == 'win32' else f'Activés ({training_args.dataloader_num_workers})'}"
    )

    # Entourer l'entraînement avec la protection de multiprocessing
    if __name__ == "__main__":
        # Ajouter freeze_support() pour les exécutables Windows
        if sys.platform == "win32":
            multiprocessing.freeze_support()

        trainer.train()

        print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 50)

        # Sauvegarder avec les métadonnées nécessaires
        output_dir = "models/trained"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        model.config.save_pretrained(output_dir)

        # Sauvegarder les métriques d'entraînement avec plus d'informations
        training_stats = {
            "final_loss": trainer.state.log_history[-1].get("loss"),
            "best_eval_loss": trainer.state.best_metric,
            "num_epochs": trainer.state.num_train_epochs,
            "total_steps": trainer.state.global_step,
            "model_type": "gpt2",
            "base_model": "dbddv01/gpt2-french-small",
            "tensorboard_log_dir": tensorboard_log_dir if has_tensorboard else None,
            "training_date": current_time,
            "dataset_size": n_samples,
            "batch_size": suggested_batch_size
            * training_args.gradient_accumulation_steps,
            "learning_rate": float(training_args.learning_rate),
            "weight_decay": float(training_args.weight_decay),
        }

        with open(f"{output_dir}/training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)

        print("\n✅ Modèle entraîné et sauvegardé dans: {output_dir}")

        if has_tensorboard:
            print(f"\nPour visualiser les métriques d'entraînement:")
            print(f"tensorboard --logdir={tensorboard_log_dir}")

except Exception as e:
    print("\n❌ ERREUR PENDANT L'ENTRAÎNEMENT")
    print("=" * 50)
    print(f"Nature de l'erreur: {str(e)}")
    import traceback

    traceback.print_exc()

# Bloc principal pour assurer que le multiprocessing fonctionne correctement
if __name__ == "__main__":
    # Si le code n'est pas déjà exécuté dans le bloc principal ci-dessus
    pass  # Tout le contenu est déjà dans le bloc try-except
