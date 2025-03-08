import os
import json
import torch
from torch.utils.data import Dataset

# 🔍 Déterminer le chemin racine du projet (comme dans train_tokenizer.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TextDataset(Dataset):
    def __init__(self, data_path, max_length=512, min_length=2):
        """
        Initialize the dataset with tokenized text data.
        
        Args:
            data_path: Path to a JSON file containing tokenized sequences
            max_length: Maximum sequence length (for padding/truncation)
            min_length: Minimum sequence length to include (default: 2 for input/target)
        """
        self.max_length = max_length
        self.min_length = min_length
        
        # Obtenir le chemin absolu à partir de la racine du projet
        absolute_data_path = os.path.join(PROJECT_ROOT, data_path)
        
        # 📂 Charger les données tokenisées
        with open(absolute_data_path, "r") as f:
            all_data = json.load(f)
            
        # Filter out sequences that are too short
        self.data = [seq for seq in all_data if len(seq) >= self.min_length]
        
        # Log if any sequences were filtered
        if len(all_data) != len(self.data):
            print(f"⚠️ Filtered out {len(all_data) - len(self.data)} sequences that were too short (< {min_length} tokens)")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the tokenized sequence
        tokens = self.data[idx]
        
        # Ensure minimum length (should not be necessary due to filtering, but just in case)
        if len(tokens) < self.min_length:
            # Pad with zeros to minimum length
            tokens = tokens + [0] * (self.min_length - len(tokens))
        
        # Truncate if longer than max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return tokens_tensor

def collate_batch(batch):
    """
    Custom collate function that pads sequences in a batch to the same length.
    Also ensures all sequences are at least length 2 for input/target pairs.
    """
    # Filter out any potentially problematic sequences (should be handled by dataset but just in case)
    valid_seqs = [seq for seq in batch if seq.size(0) >= 2]
    
    # If no valid sequences, return a minimal valid tensor to avoid crashes
    if not valid_seqs:
        return torch.zeros((1, 2), dtype=torch.long)  # Return minimal valid batch
    
    # Find the max length in this batch
    max_len = max([seq.size(0) for seq in valid_seqs])
    
    # Create the output tensor and fill with padding token (usually 0)
    batch_size = len(valid_seqs)
    padded_batch = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # Copy data to output tensor
    for i, seq in enumerate(valid_seqs):
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
