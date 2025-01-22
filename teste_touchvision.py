import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")


print(torch.cuda.is_available())  # Deve retornar True
print(torch.version.cuda)  # Deve retornar '11.8'