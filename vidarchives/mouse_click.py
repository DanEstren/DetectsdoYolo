import cv2
import numpy as np
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordenadas do clique: ({x}, {y})")

# Caminho do vídeo (substitua pelo caminho do seu vídeo)
video_path = 'videos/vacas.mp4'
RESIZE_SCALE = 0.7
# Abre o vídeo
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)

counting_area = np.array([(15, 459), (32, 853), (120, 847), (103, 457)])


# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Obtém o total de frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total de frames no vídeo: {total_frames}")

# Pergunta ao usuário qual frame deseja visualizar
while True:
    try:
        frame_number = int(input(f"Digite o número do frame (0 a {total_frames-1}): "))
        if 0 <= frame_number < total_frames:
            break
        else:
            print(f"Por favor, insira um número entre 0 e {total_frames-1}.")
    except ValueError:
        print("Por favor, insira um número válido.")

# Define o frame desejado
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Lê o frame especificado
ret, frame = cap.read()
if not ret:
    print("Erro ao ler o frame")
    cap.release()
    exit()

# Cria uma janela e configura o callback do mouse
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

# Exibe o frame
while True:
    frame = cv2.resize(frame, (width, height))
    cv2.polylines(frame, pts=[counting_area], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imshow('Frame', frame)
    
    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()