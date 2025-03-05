import torch

print(torch.__version__)  # Vérifie la version installée
print(torch.version.cuda)  # Doit afficher une version de CUDA (ex: "12.1")
print(torch.cuda.is_available())  # Doit être True
print(torch.cuda.get_device_name(0))  # Doit afficher "NVIDIA GeForce RTX 4050"
