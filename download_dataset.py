from roboflow import Roboflow
from config import ROBOFLOW_API_KEY
import os

# Download do dataset customizado do Roboflow, este dataset foi criado baseado (fork) do coco128
# https://universe.roboflow.com/objectdetectv1-03u2h/object-detection-v2-dm7ea/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

# Configuração da API do Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# versão do dataset a ser baixada   
dataset_version = 2

# Caminho da pasta de destino
current_directory = os.getcwd()
destination_folder = f"{current_directory}\\datasets\\dataset-version-{dataset_version}" 

# Criar a pasta "datasets" se ela não existir
os.makedirs(destination_folder, exist_ok=True)

project = rf.workspace("objectdetectv1-03u2h").project("object-detection-v2-dm7ea")
version = project.version(dataset_version)
dataset = version.download("yolov11", location=destination_folder, overwrite=True)