from ultralytics import YOLO
import shutil
import os
import torch

def main():
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {_device}")
    
    # versão do dataset a ser baixada   
    dataset_version = 2

    # obs: faça o download do dataset customizado primeiro
    current_directory = os.getcwd()
    data_path = f"{current_directory}\\datasets\\dataset-version-{dataset_version}\\data.yaml"  

    # Criar e configurar o modelo
    model = YOLO("yolo11x.pt")
    model.to(_device)

    # Treinamento do modelo
    model.train(
        data=data_path,     # Arquivo .yaml com a configuração do dataset
        epochs=10,          # Número de épocas para treinamento
        imgsz=640,          # Tamanho das imagens
        batch=8,            # Tamanho do batch
        plots=True,         # Gera gráficos do processo de treinamento
        device=_device,     # Dispositivo a ser utilizado
        cache="ram",        # Cache na RAM
        freeze=[0, 24],     # Congelar as camadas iniciais
        workers=4,          # Número de threads de carregamento de dados
        project="output/custom_model"  # Nome do diretório base
    )

    # Copia o modelo treinado para um novo diretório
    original_path = f"{current_directory}\\output\\custom_model\\train\\weights\\best.pt"
    new_path = f"{current_directory}\\yolo11x-custom.pt"
    shutil.copy(original_path, new_path)

if __name__ == "__main__":
    main()
