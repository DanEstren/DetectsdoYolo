import cv2
import json
from ultralytics import YOLO
from collections import defaultdict

# Carregar o modelo YOLOv8
model = YOLO("yolo11x.pt").to(device="cuda")  # Use yolov8n.pt (nano) ou outro modelo, como yolov8s.pt, yolov8m.pt, etc.

# Caminho do vídeo de entrada
video_path = "videos/futebol.mp4"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Dicionário para armazenar informações de rastreamento
tracks = defaultdict(lambda: {"first_frame": None, "last_frame": None, "first_box": None, "last_box": None})

# Configurações do vídeo
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Loop para processar o vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Realizar detecção e rastreamento com YOLOv8
    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml")  # Classe 0 = pessoa

    # Processar os resultados
    for result in results:
        if result.boxes.id is not None:  # Verificar se há IDs de rastreamento
            boxes = result.boxes.xyxy.cpu().numpy()  # Caixas delimitadoras (x_min, y_min, x_max, y_max)
            ids = result.boxes.id.cpu().numpy().astype(int)  # IDs das pessoas

            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                box = box.tolist()  # Converter para lista para serialização JSON

                # Registrar a primeira aparição
                if tracks[track_id]["first_frame"] is None:
                    tracks[track_id]["first_frame"] = frame_count
                    tracks[track_id]["first_box"] = box

                # Atualizar a última aparição
                tracks[track_id]["last_frame"] = frame_count
                tracks[track_id]["last_box"] = box

    # Opcional: Exibir o frame com detecções (para depuração)
    annotated_frame = results[0].plot()  # Desenhar caixas e IDs no frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Salvar os resultados em um arquivo JSON
output_data = {
    "video_info": {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_count
    },
    "tracks": {
        f"person_{track_id}": {
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "first_box": info["first_box"],
            "last_box": info["last_box"]
        }
        for track_id, info in tracks.items()
    }
}

with open("outputs/tracking_results.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Rastreamento concluído. Resultados salvos em 'tracking_results.json'.")