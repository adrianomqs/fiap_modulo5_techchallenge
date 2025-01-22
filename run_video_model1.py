from ultralytics import YOLO
import torch

# Verificar se a GPU está disponível
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {_device}")

# Habilitar o uso de um modelo customizado
model_custom = True

# seleciona o modelo a ser carregado
if model_custom:
    # Load a custom YOLOv5 model
    model_name = "yolo11x-custom.pt"
else:
    # Load a COCO-pretrained YOLO11n model
    model_name = "yolo11x.pt"

# Load a COCO-pretrained YOLO11n model
model = YOLO(model_name)
model.to(_device)

results = model.track("videos/video.mp4", save=True, show=True)