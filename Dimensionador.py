import cv2
import numpy as np

# === CONFIGURAÇÕES ===
RTSP_URL = 0
LARGURA_DA_JANELA = 1024 # Tamanho confortável para ver no monitor

# Variáveis globais
pontos_reais = [] # Salva as coordenadas da imagem ORIGINAL (Gigante)
pontos_visual = [] # Salva as coordenadas da telinha (só pra desenhar)

def mouse_callback(event, x, y, flags, param):
    global pontos_reais, pontos_visual
    
    # Param contém as proporções (scale_x, scale_y) que passamos na criação do callback
    scale_w, scale_h = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Pega o clique na tela pequena (x, y)
        pontos_visual.append((x, y))
        
        # 2. Converte para a coordenada REAL (Regra de 3)
        x_real = int(x * scale_w)
        y_real = int(y * scale_h)
        pontos_reais.append((x_real, y_real))
        
        print(f"✅ Clique Tela: ({x}, {y}) -> 🌍 Real: ({x_real}, {y_real})")
        print(f"📋 Lista para copiar: {pontos_reais}\n")

    elif event == cv2.EVENT_RBUTTONDOWN:
        pontos_reais = []
        pontos_visual = []
        print("🗑️ Pontos limpos!")

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Erro ao abrir câmera!")
        return

    # 1. Descobre o tamanho REAL da imagem da câmera
    # Geralmente é 1920x1080 ou 1280x720
    w_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📏 Resolução Original da Câmera: {w_original}x{h_original}")

    # 2. Calcula o tamanho da JANELA (Mantendo a proporção correta)
    # Se a largura for 1024, qual deve ser a altura para não esticar?
    fator_proporcao = LARGURA_DA_JANELA / w_original
    w_tela = LARGURA_DA_JANELA
    h_tela = int(h_original * fator_proporcao)
    
    print(f"🖥️  Resolução da Janela Visual: {w_tela}x{h_tela}")
    print(f"➗ Fator de Escala: {1/fator_proporcao:.2f}x")

    # 3. Calcula os multiplicadores para corrigir o clique
    scale_w = w_original / w_tela
    scale_h = h_original / h_tela

    window_name = "Seletor de ROI Inteligente"
    cv2.namedWindow(window_name)
    # Passamos os fatores de escala para a função do mouse saber calcular
    cv2.setMouseCallback(window_name, mouse_callback, param=(scale_w, scale_h))

    print("\n--- INSTRUÇÕES ---")
    print("🖱️  Botão ESQUERDO: Marcar ponto")
    print("🖱️  Botão DIREITO:  Resetar")
    print("⌨️  'Q': Sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tentando reconectar...")
            continue

        # Redimensiona apenas para VISUALIZAÇÃO
        frame_visual = cv2.resize(frame, (w_tela, h_tela))

        # Desenha os pontos (usando as coordenadas da tela pequena)
        if len(pontos_visual) > 0:
            # Desenha bolinhas
            for p in pontos_visual:
                cv2.circle(frame_visual, p, 5, (0, 0, 255), -1)
            
            # Desenha linhas
            if len(pontos_visual) > 1:
                pts = np.array(pontos_visual, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_visual, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, frame_visual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ao sair, imprime o código pronto para você
    print("\n" + "="*40)
    print("🚀 COPIE E COLE ISSO NO SEU CÓDIGO:")
    print(f"roi_points = np.array({pontos_reais}, np.int32)")
    print("="*40)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
