import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Config
from dataset import TextDataset, collate_batch
from model import CustomGPT
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import numpy as np
from tqdm import tqdm
import sys

# 📂 Chemins des fichiers
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DATA_PATH = "data/tokenized/train.json"
VAL_DATA_PATH = "data/tokenized/val.json"
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_custom.pth")

# 📈 Configuration de TensorBoard
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(PROJECT_ROOT, "logs", f"Sushi_AI_{timestamp}")
os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
writer = SummaryWriter(log_dir)
print(f"📊 TensorBoard logs will be saved to {log_dir}")
print(
    f"📊 Start TensorBoard with: tensorboard --logdir={os.path.join(PROJECT_ROOT, 'logs')}"
)

# 📝 Fonction pour enregistrer les logs dans un fichier texte
log_file = os.path.join(log_dir, "training_log.txt")


def log_message(message):
    """Write log message to console and log file using UTF-8 encoding"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# 📌 Hyperparamètres
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 📊 Afficher les informations système
log_message(f"Device: {DEVICE}")  # Removed emoji for troubleshooting
if torch.cuda.is_available():
    log_message(f"GPU: {torch.cuda.get_device_name(0)}")
    log_message(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB"
    )
log_message(f"PyTorch version: {torch.__version__}")

# 🔥 Charger les datasets
train_dataset = TextDataset(TRAIN_DATA_PATH)
val_dataset = TextDataset(VAL_DATA_PATH)

# Use custom collate function to handle variable sequence lengths
# Define DataLoaders BEFORE referencing them in log messages
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
)

log_message(f"Train dataset: {len(train_dataset)} samples")
log_message(f"Validation dataset: {len(val_dataset)} samples")
log_message(
    f"Batch size: {BATCH_SIZE} (training batches: {len(train_dataloader)}, validation batches: {len(val_dataloader)})"
)

# 📌 Initialiser le modèle
config = GPT2Config(
    vocab_size=32000,
    n_embd=768,        # Taille des embeddings (augmentée de 512 à 768)
    n_layer=12,        # Nombre de couches (augmenté de 6 à 12)
    n_head=12,         # Nombre de têtes d'attention (augmenté de 8 à 12)
)
model = CustomGPT(config).to(DEVICE)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log_message(f"🧠 Model parameters: {total_params:,} (~{total_params/1_000_000:.1f}M) (trainable: {trainable_params:,})")

# 📌 Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Log model architecture and hyperparameters
writer.add_text(
    "Model/Architecture",
    f"GPT-2 with {config.n_layer} layers, {config.n_head} attention heads, {total_params:,} parameters",
)
writer.add_text("Hyperparameters/batch_size", str(BATCH_SIZE))
writer.add_text("Hyperparameters/learning_rate", str(LEARNING_RATE))
writer.add_text("Hyperparameters/epochs", str(EPOCHS))
writer.add_text("System/device", DEVICE)
if torch.cuda.is_available():
    writer.add_text("System/gpu", torch.cuda.get_device_name(0))


# 🔄 Fonction pour visualiser quelques prédictions dans TensorBoard
def log_predictions(model, dataloader, tokenizer_path, step, writer, n_samples=2):
    """Generate and log text predictions to TensorBoard"""
    try:
        # Pas besoin d'importer le tokenizer si on veut juste logger les IDs
        # Cette fonction peut être améliorée si vous avez accès au tokenizer
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(DEVICE)
                input_seq = batch[:n_samples, :20]  # Prendre les 20 premiers tokens

                # Log simplement les IDs (en attendant d'accéder au tokenizer)
                for i in range(min(n_samples, input_seq.shape[0])):
                    sample = input_seq[i]
                    writer.add_text(
                        f"Samples/input_seq_{i}", str(sample.tolist()), step
                    )
                break
    except Exception as e:
        log_message(f"⚠️ Error logging predictions: {str(e)}")


# 📊 Fonction pour mesurer la perplexité (metric standard pour les LLMs)
def calculate_perplexity(loss):
    """Calculate perplexity from cross entropy loss"""
    return torch.exp(torch.tensor(loss)).item()


# 📌 Fonction d'évaluation
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(DEVICE)
            
            # Skip batch if too short (needs at least 2 tokens: 1 for input, 1 for target)
            if batch.size(1) < 2:
                log_message(f"⚠️ Skipping batch with too short sequence: {batch.size()}")
                continue
            
            # 🔄 Décaler les tokens pour l'apprentissage (comme dans la boucle d'entraînement)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Double check that input_ids are not empty
            if input_ids.numel() == 0:
                log_message(f"⚠️ Empty input_ids detected in validation, skipping batch")
                continue

            outputs = model(input_ids)
            
            outputs_flat = outputs.reshape(-1, config.vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Check if we have any valid targets
            if targets_flat.numel() == 0:
                log_message(f"⚠️ Empty targets detected in validation, skipping batch")
                continue
                
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item() * targets_flat.numel()
            total_tokens += targets_flat.numel()

    # Guard against division by zero if all batches were skipped
    if total_tokens == 0:
        log_message("⚠️ No valid tokens processed in validation. Check your data!")
        return {"loss": float('inf'), "perplexity": float('inf'), "time": time.time() - start_time}
            
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    eval_time = time.time() - start_time

    return {"loss": avg_loss, "perplexity": perplexity, "time": eval_time}


# 📈 Variables pour le tracking des métriques
best_val_loss = float("inf")
train_loss_history = []
val_loss_history = []
lr_history = []
time_history = []

# 📌 Boucle d'entraînement
log_message("\n" + "=" * 50)
log_message(f"🚀 Starting training: {EPOCHS} epochs")
log_message("=" * 50)

total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    total_tokens = 0
    step = 0

    # Rolling metrics for logging
    rolling_train_loss = []
    rolling_window = 50

    # Create progress bar
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for batch in progress_bar:
        step_start = time.time()
        batch = batch.to(DEVICE)

        # 🔄 Décaler les tokens pour l'apprentissage
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()

        # Get model outputs
        outputs = model(input_ids)

        # Reshape outputs and targets for loss calculation
        outputs_flat = outputs.reshape(-1, config.vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate loss
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer.step()

        # Update metrics
        current_loss = loss.item()
        rolling_train_loss.append(current_loss)
        if len(rolling_train_loss) > rolling_window:
            rolling_train_loss.pop(0)

        avg_rolling_loss = np.mean(rolling_train_loss)
        perplexity = calculate_perplexity(avg_rolling_loss)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{avg_rolling_loss:.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            }
        )

        # Count tokens for weighted loss
        num_tokens = targets.numel()
        total_loss += current_loss * num_tokens
        total_tokens += num_tokens

        # Log batch metrics
        global_step = epoch * len(train_dataloader) + step
        writer.add_scalar("Training/BatchLoss", current_loss, global_step)
        writer.add_scalar("Training/RollingLoss", avg_rolling_loss, global_step)
        writer.add_scalar("Training/Perplexity", perplexity, global_step)
        writer.add_scalar(
            "Training/LearningRate", optimizer.param_groups[0]["lr"], global_step
        )

        # Log GPU memory usage if available
        if torch.cuda.is_available():
            writer.add_scalar(
                "System/GPU_Memory_Used_GB",
                torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024,
                global_step,
            )
            writer.add_scalar(
                "System/GPU_Memory_Cached_GB",
                torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024,
                global_step,
            )

        # Log step timing
        step_time = time.time() - step_start
        writer.add_scalar("System/StepTimeSeconds", step_time, global_step)

        step += 1

    # Calculate weighted average training loss for this epoch
    avg_train_loss = total_loss / total_tokens
    train_loss_history.append(avg_train_loss)
    train_perplexity = calculate_perplexity(avg_train_loss)

    # Evaluate on validation set
    log_message(f"\n🔍 Evaluating on validation set...")
    val_metrics = evaluate(model, val_dataloader, criterion)
    val_loss_history.append(val_metrics["loss"])

    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    time_history.append(epoch_time)

    # Log sample predictions
    log_predictions(model, val_dataloader, None, epoch, writer)

    # Log metrics for this epoch
    writer.add_scalar("Training/EpochLoss", avg_train_loss, epoch)
    writer.add_scalar("Training/EpochPerplexity", train_perplexity, epoch)
    writer.add_scalar("Validation/Loss", val_metrics["loss"], epoch)
    writer.add_scalar("Validation/Perplexity", val_metrics["perplexity"], epoch)
    writer.add_scalar("System/EpochTimeMinutes", epoch_time / 60, epoch)

    # Log model parameter histograms
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"Parameters/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    # Check if this is the best model
    is_best = val_metrics["loss"] < best_val_loss
    if is_best:
        best_val_loss = val_metrics["loss"]

    # Log to console
    log_message(
        f"\n📊 Epoch {epoch+1}/{EPOCHS}:\n"
        f"   Train Loss: {avg_train_loss:.4f}, Perplexity: {train_perplexity:.2f}\n"
        f"   Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}\n"
        f"   Time: {epoch_time/60:.2f} min (train: {epoch_time - val_metrics['time']:.1f}s, val: {val_metrics['time']:.1f}s)\n"
        f"   {'✅ New best model!' if is_best else ''}"
    )

    # 🔥 Sauvegarde du modèle après chaque epoch
    model_save_path = os.path.join(
        PROJECT_ROOT, "models", f"gpt2_custom_epoch{epoch+1}.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "train_perplexity": train_perplexity,
            "val_perplexity": val_metrics["perplexity"],
            "best_model": is_best,
        },
        model_save_path,
    )
    log_message(f"💾 Checkpoint saved to {model_save_path}")

    # Also save best model separately
    if is_best:
        best_model_path = os.path.join(PROJECT_ROOT, "models", "gpt2_custom_best.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["loss"],
                "train_perplexity": train_perplexity,
                "val_perplexity": val_metrics["perplexity"],
            },
            best_model_path,
        )
        log_message(f"🏆 New best model saved to {best_model_path}")

# Log final training curve
fig_data = [[x, y] for (x, y) in zip(range(EPOCHS), train_loss_history)]
writer.add_custom_scalars(
    {
        "Training": {
            "Loss Over Time": ["Multiline", ["Training/EpochLoss", "Validation/Loss"]],
            "Perplexity Over Time": [
                "Multiline",
                ["Training/EpochPerplexity", "Validation/Perplexity"],
            ],
        }
    }
)

# Calculate total training time
total_time = time.time() - total_start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

# Close the TensorBoard writer
writer.close()

# Final stats
log_message("\n" + "=" * 50)
log_message(f"✅ Training completed in {hours}h {minutes}m {seconds}s")
log_message(f"🏆 Best validation loss: {best_val_loss:.4f}")
log_message(f"📊 TensorBoard logs available at {log_dir}")
log_message(f"📝 Training log saved to {log_file}")
log_message("=" * 50)
