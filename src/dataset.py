import os
import json
import torch
from torch.utils.data import Dataset

# 🔍 Déterminer le chemin racine du projet (comme dans train_tokenizer.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        
        # Obtenir le chemin absolu à partir de la racine du projet
        absolute_data_path = os.path.join(PROJECT_ROOT, data_path)
        
        # 📂 Charger les données tokenisées
        with open(absolute_data_path, "r") as f:
            raw_data = json.load(f)
        
        # 🔄 Transformer chaque séquence en tensor PyTorch
        for sequence in raw_data:
            self.data.append(torch.tensor(sequence, dtype=torch.long))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
