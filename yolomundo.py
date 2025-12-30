from ultralytics import YOLOWorld
import cv2

RTSP_URL = "rtsp://admin:Loja0500@@201.48.213.113:80/cam/realmonitor?channel=1&subtype=0" 

# Carrega o modelo
model = YOLOWorld("yolov8x-world.pt")

# DICA: No YOLO-World, defina o que você quer buscar, senão ele pode não achar nada
# ou usar as classes padrão do COCO. Exemplo:
model.set_classes(["tape", "pen", "adhesive tape", "clipboard", "cup"]) 

cap = cv2.VideoCapture(RTSP_URL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame ou stream acabou.")
        break
    
    # Faz a predição. verbose=False evita encher o terminal de texto
    results = model.predict(frame, verbose=False)
    
    # AQUI ESTÁ O SEGREDO:
    # 1. Pega o primeiro resultado (results[0])
    # 2. Usa .plot() para desenhar as caixas e gerar uma imagem numpy (array)
    annotated_frame = results[0].plot()

    # Agora sim, o imshow recebe uma imagem válida
    cv2.imshow("video", annotated_frame)

    # Sai se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()