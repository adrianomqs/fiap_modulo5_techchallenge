from roboflow import Roboflow
from config import ROBOFLOW_API_KEY
import os

# Configuração da API do Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Caminho da pasta de destino
destination_folder = "datasets" 

# Criar a pasta "datasets" se ela não existir
os.makedirs(destination_folder, exist_ok=True)

project = rf.workspace("objectdetectv1-03u2h").project("object-detection-dataset-bmpr6")
version = project.version(2)
dataset = version.download("yolov11", location=destination_folder)