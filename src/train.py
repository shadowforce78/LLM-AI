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

# 📂 Chemins des fichiers
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tokenized", "train.json")
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tokenized", "val.json")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_custom.pth")

# 📈 Configuration de TensorBoard
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(PROJECT_ROOT, "logs", f"Sushi_AI_{timestamp}")
writer = SummaryWriter(log_dir)
print(f"📊 TensorBoard logs will be saved to {log_dir}")
print(f"📊 Start TensorBoard with: tensorboard --logdir={os.path.join(PROJECT_ROOT, 'logs')}")

# 📌 Hyperparamètres
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Charger les datasets
train_dataset = TextDataset(TRAIN_DATA_PATH)
val_dataset = TextDataset(VAL_DATA_PATH)

# Use custom collate function to handle variable sequence lengths
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# 📌 Initialiser le modèle
config = GPT2Config(
    vocab_size=32000,
    n_embd=512,
    n_layer=6,
    n_head=8,
)
model = CustomGPT(config).to(DEVICE)

# 📌 Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Log model architecture and hyperparameters
writer.add_text("Model/Architecture", f"GPT-2 with {config.n_layer} layers, {config.n_head} attention heads")
writer.add_text("Hyperparameters/batch_size", str(BATCH_SIZE))
writer.add_text("Hyperparameters/learning_rate", str(LEARNING_RATE))
writer.add_text("Hyperparameters/epochs", str(EPOCHS))

# 📌 Fonction d'évaluation
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            outputs = model(input_ids)
            
            outputs_flat = outputs.reshape(-1, config.vocab_size)
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 📌 Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    step = 0

    for batch in train_dataloader:
        batch = batch.to(DEVICE)

        # 🔄 Décaler les tokens pour l'apprentissage
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        
        # Get model outputs - shape [batch_size, seq_len, vocab_size]
        outputs = model(input_ids)
        
        # Reshape outputs and targets for loss calculation
        outputs_flat = outputs.reshape(-1, config.vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Log batch loss every 10 steps
        if step % 10 == 0:
            global_step = epoch * len(train_dataloader) + step
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], global_step)
        
        step += 1

    # Calculate average training loss for this epoch
    avg_train_loss = total_loss / len(train_dataloader)
    
    # Evaluate on validation set
    avg_val_loss = evaluate(model, val_dataloader, criterion)
    
    # Log metrics for this epoch
    writer.add_scalar('Training/EpochLoss', avg_train_loss, epoch)
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    
    # Log model parameter histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f"Parameters/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
    
    print(f"📈 Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # 🔥 Sauvegarde du modèle après chaque epoch
    model_save_path = os.path.join(PROJECT_ROOT, "models", f"gpt2_custom_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, model_save_path)
    print(f"💾 Checkpoint saved to {model_save_path}")

# Close the TensorBoard writer
writer.close()

print(f"✅ Modèle entraîné et sauvegardé dans {MODEL_SAVE_PATH}")
print(f"📊 TensorBoard logs disponibles dans {log_dir}")
