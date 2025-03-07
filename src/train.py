import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Config
from dataset import TextDataset
from model import CustomGPT

# 📂 Chemins des fichiers
TRAIN_DATA_PATH = "data/tokenized/train.json"
VAL_DATA_PATH = "data/tokenized/val.json"
MODEL_SAVE_PATH = "models/trained/gpt-small.pth"

# 📌 Hyperparamètres
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Charger les datasets
train_dataset = TextDataset(TRAIN_DATA_PATH)
val_dataset = TextDataset(VAL_DATA_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# 📌 Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = batch.to(DEVICE)

        # 🔄 Décaler les tokens pour l'apprentissage
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_train_loss:.4f}")

    # 🔥 Sauvegarde du modèle après chaque epoch
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"✅ Modèle entraîné et sauvegardé dans {MODEL_SAVE_PATH}")
