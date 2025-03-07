import os
import json
import torch
from torch.utils.data import Dataset

# 🔍 Déterminer le chemin racine du projet (comme dans train_tokenizer.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TextDataset(Dataset):
    def __init__(self, data_path, max_length=512):
        """
        Initialize the dataset with tokenized text data.
        
        Args:
            data_path: Path to a JSON file containing tokenized sequences
            max_length: Maximum sequence length (for padding/truncation)
        """
        self.max_length = max_length
        
        # Obtenir le chemin absolu à partir de la racine du projet
        absolute_data_path = os.path.join(PROJECT_ROOT, data_path)
        
        # 📂 Charger les données tokenisées
        with open(absolute_data_path, "r") as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the tokenized sequence
        tokens = self.data[idx]
        
        # Truncate if longer than max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return tokens_tensor

def collate_batch(batch):
    """
    Custom collate function that pads sequences in a batch to the same length
    """
    # Find the max length in this batch
    max_len = max([seq.size(0) for seq in batch])
    
    # Create the output tensor and fill with padding token (usually 0)
    batch_size = len(batch)
    padded_batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # Copy data to output tensor
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded_batch[i, :seq_len] = seq
        
    return padded_batch

# 🛠 Test rapide si exécuté directement
if __name__ == "__main__":
    # Chemins relatifs à partir de la racine du projet (comme dans train_tokenizer.py)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "tokenized")
    train_file = os.path.join("data", "tokenized", "train.json")
    
    dataset = TextDataset(train_file)
    print(f"✅ Dataset chargé avec {len(dataset)} échantillons.")
    print(f"🔍 Exemple : {dataset[0][:10]}")  # Affiche les 10 premiers tokens du premier batch
    print(f"📁 Racine du projet: {PROJECT_ROOT}")
    print(f"📁 Dossier des données: {DATA_DIR}")
