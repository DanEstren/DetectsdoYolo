import json
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Função para desenhar caixas delimitadoras com IDs
def draw_boxes(frame, boxes, ids):
    for box, track_id in zip(boxes, ids):
        x_min, y_min, x_max, y_max = map(int, box)
        # Desenhar retângulo (caixa delimitadora)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Adicionar ID acima da caixa
        cv2.putText(frame, f"ID: {track_id}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Caminho do vídeo e do arquivo JSON
video_path = "./videos/futebol.mp4"  # Substitua pelo caminho do seu vídeo
json_path = "./outputs/tracking_results.json"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Ler o arquivo JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Extrair informações dos tracks
tracks = data["tracks"]

# Receber o número do frame como entrada
print(f"Digite o numero do Frame que deseja detectar de {total_frames} ")
frame_number = int(input("> : "))

# Identificar jogadores presentes no frame especificado
active_players = []
active_boxes = []
active_ids = []
for player_id, info in tracks.items():
    first_frame = info["first_frame"]
    last_frame = info["last_frame"]
    if first_frame <= frame_number <= last_frame:
        # Usar a caixa do primeiro ou último frame, dependendo da proximidade
        box = info["first_box"] if abs(frame_number - first_frame) <= abs(frame_number - last_frame) else info["last_box"]
        active_players.append(player_id)
        active_boxes.append(box)
        active_ids.append(player_id.split("_")[1])  # Extrair número do ID (ex.: "1" de "person_1")

# Verificar se há jogadores no frame
if not active_players:
    print(f"Nenhum jogador detectado no frame {frame_number}.")
    exit()

# Definir o frame desejado
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Frames são indexados a partir de 0
ret, frame = cap.read()
if not ret:
    print(f"Erro ao carregar o frame {frame_number}.")
    cap.release()
    exit()

# Desenhar caixas delimitadoras no frame
frame = draw_boxes(frame, active_boxes, active_ids)

# Converter frame de BGR (OpenCV) para RGB (Matplotlib)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Exibir e salvar o frame usando Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(frame_rgb)
plt.axis('off')  # Remover eixos
plt.title(f"Frame {frame_number} com Caixas Delimitadoras")
plt.tight_layout()

# Salvar a imagem
output_path = f"frame_{frame_number}_boxes.png"
plt.savefig(output_path)
plt.close()  # Fechar a figura para liberar memória
cap.release()

print(f"Imagem salva como '{output_path}'.")