from ultralytics import YOLO
import cv2
import torch

# Verificar se a GPU está disponível
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {_device}")

#_device = "cpu"

# Especifica a confiança mínima e as classes de detecção
_classes = [43, 73]  # Substitua se necessário
_conf = 0

# Carregar o modelo e mover para o dispositivo
model = YOLO("yolo11x.pt")  # Substitua pelo caminho do modelo
model.to(_device)

# Abrir o vídeo
cap = cv2.VideoCapture("videos/video.mp4")
if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo!")
    exit()

out = cv2.VideoWriter("output/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
if not out.isOpened():
    print("Erro ao abrir o arquivo de saída de vídeo!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção apenas para facas (classe = 43)
    results = model.predict(frame, device=_device, classes=_classes) #, conf=_conf)

    # Processar os resultados e desenhar caixas
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = map(float, box)
        if int(class_id) in _classes:
            label = f"Knife: {conf:.2f}"

            # Desenhar a caixa e a etiqueta no frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Escrever o frame processado no vídeo de saída
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
