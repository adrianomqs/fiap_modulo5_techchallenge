from roboflow import Roboflow
from config import ROBOFLOW_API_KEY
import os

# Configuração da API do Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# versão do dataset a ser baixada   
dataset_version = 4

# Caminho da pasta de destino
current_directory = os.getcwd()
destination_folder = f"{current_directory}\\datasets\\dataset-version-{dataset_version}" 

# Criar a pasta "datasets" se ela não existir
os.makedirs(destination_folder, exist_ok=True)

project = rf.workspace("objectdetectv1-03u2h").project("object-detection-dataset-bmpr6")
version = project.version(dataset_version)
dataset = version.download("yolov11", location=destination_folder, overwrite=True)