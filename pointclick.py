import cv2
import numpy as np

# Lista para armazenar os pontos clicados
pontos = []
frame_display = None

def mouse_callback(event, x, y, flags, param):
    global pontos, frame_display
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique com botão esquerdo
        # Adiciona o ponto à lista
        pontos.append((x, y))
        print(f"Ponto {len(pontos)}: X={x}, Y={y}")
        
        # Cria uma cópia do frame para desenhar
        frame_display = param.copy()
        
        # Desenha todos os pontos
        for i, ponto in enumerate(pontos):
            cv2.circle(frame_display, ponto, 5, (0, 255, 0), -1)
            # Adiciona o número do ponto
            cv2.putText(frame_display, str(i+1), 
                       (ponto[0]+10, ponto[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Desenha linhas conectando os pontos
        if len(pontos) > 1:
            for i in range(len(pontos) - 1):
                cv2.line(frame_display, pontos[i], pontos[i+1], (255, 0, 0), 2)
        
        # Atualiza a janela
        cv2.imshow('Frame - Clique para marcar pontos', frame_display)

# URL ou caminho do vídeo
video_url = "videos/bus.mov"  # Substitua pelo caminho do seu vídeo

# Captura o vídeo
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Lê o primeiro frame
ret, frame = cap.read()

if not ret:
    print("Erro ao ler o frame!")
    cap.release()
    exit()

# Cria uma cópia do frame original
frame_original = frame.copy()
frame_display = frame.copy()

# Cria a janela e configura o callback do mouse
cv2.namedWindow('Frame - Clique para marcar pontos')
cv2.setMouseCallback('Frame - Clique para marcar pontos', mouse_callback, frame_original)

# Mostra o frame
cv2.imshow('Frame - Clique para marcar pontos', frame_display)

print("\nInstruções:")
print("- Clique com o botão esquerdo para marcar pontos")
print("- Pressione 'r' para resetar os pontos")
print("- Pressione 'q' para sair")
print("- Pressione 's' para salvar a imagem\n")

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Sair
        break
    elif key == ord('r'):  # Resetar pontos
        pontos = []
        frame_display = frame_original.copy()
        cv2.imshow('Frame - Clique para marcar pontos', frame_display)
        print("\nPontos resetados!\n")
    elif key == ord('s'):  # Salvar imagem
        cv2.imwrite('frame_marcado.png', frame_display)
        print("\nImagem salva como 'frame_marcado.png'\n")

# Libera recursos
cap.release()
cv2.destroyAllWindows()

# Mostra todos os pontos marcados
print("\n=== Pontos Marcados ===")
for i, (x, y) in enumerate(pontos):
    print(f"Ponto {i+1}: X={x}, Y={y}")