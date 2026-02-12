import cv2
import numpy as np

# === CONFIGURAÇÕES ===
# Coloque sua URL aqui
RTSP_URL = 0

# depois usar assim

# roi_points = np.array([(270, 6), (289, 1075), (1675, 1074), (1566, 7)], np.int32)

# x_corte, y_corte, w_corte, h_corte = cv2.boundingRect(roi_points)

# latest_frame = frame[y_corte : y_corte + h_corte, x_corte : x_corte + w_corte], pasar o frame para ser isso, que ele vai saber o dimensionamento correto

# TAMANHO QUE VOCÊ QUER VER NA TELA (Display)
LARGURA_DISPLAY = 1280
ALTURA_DISPLAY = 720

# Variáveis globais para guardar os pontos
pontos_reais = []   # Coordenadas da Câmera Original
pontos_visuais = [] # Coordenadas da Telinha (1280x720)

def mouse_callback(event, x, y, flags, param):
    """
    Função que processa o clique.
    'param' traz as escalas (scale_x, scale_y)
    """
    global pontos_reais, pontos_visuais
    scale_x, scale_y = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Guarda o ponto onde você clicou (Visual)
        pontos_visuais.append((x, y))
        
        # 2. Calcula onde é esse ponto na imagem REAL
        x_real = int(x * scale_x)
        y_real = int(y * scale_y)
        pontos_reais.append((x_real, y_real))
        
        print(f"🖱️ Clique: ({x}, {y}) --> 🎯 Real: ({x_real}, {y_real})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        pontos_reais = []
        pontos_visuais = []
        print("🗑️ Pontos limpos!")

def main():
    print("Conectando na câmera...")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir câmera!")
        return

    # 1. Pega a resolução REAL da câmera (sem chutar)
    w_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📏 Câmera Original: {w_original}x{h_original}")
    print(f"🖥️  Sua Tela de Desenho: {LARGURA_DISPLAY}x{ALTURA_DISPLAY}")

    # 2. Calcula o Fator de Escala (Quantas vezes a original é maior que a tela)
    # Se a câmera for 1920 e a tela 1280, o fator é 1.5
    scale_x = w_original / LARGURA_DISPLAY
    scale_y = h_original / ALTURA_DISPLAY

    print(f"➗ Fatores de Conversão: X={scale_x:.2f}, Y={scale_y:.2f}")

    # Configura Janela e Mouse
    window_name = "Definir ROI (Redimensionado)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=(scale_x, scale_y))

    print("\n--- COMANDOS ---")
    print("🖱️  Botão ESQUERDO: Marcar")
    print("🖱️  Botão DIREITO:  Limpar")
    print("⌨️  'Q': Sair e Pegar Código")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconectando...")
            continue

        # 3. FORÇA O RESIZE PARA O TAMANHO QUE VOCÊ ESCOLHEU
        frame_display = cv2.resize(frame, (LARGURA_DISPLAY, ALTURA_DISPLAY))

        # Desenha os pontos (usando as coordenadas visuais)
        if len(pontos_visuais) > 0:
            for p in pontos_visuais:
                cv2.circle(frame_display, p, 5, (0, 0, 255), -1)
            
            if len(pontos_visuais) > 1:
                pts = np.array(pontos_visuais, np.int32).reshape((-1, 1, 2))
                # isClosed=True fecha o retângulo automaticamente
                fechar = True if len(pontos_visuais) >= 3 else False
                cv2.polylines(frame_display, [pts], isClosed=fechar, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # RESULTADO FINAL
    print("\n" + "="*50)
    print("✅ PRONTO! Use esta linha no seu código principal:")
    print(f"roi_points = np.array({pontos_reais}, np.int32)")
    print("="*50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
