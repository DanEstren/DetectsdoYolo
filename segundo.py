import cv2
import numpy as np
from ultralytics import YOLO

# Carregar o modelo YOLO treinado
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\MaturidadeDoCafeYolo11X100ep3\weights\best.pt")
print("Classes do modelo:", model.names)

# Definir as classes
class_names = ['dry', 'overripe', 'ripe', 'semi_ripe', 'unripe']
if class_names != list(model.names.values()):
    print("Erro: As classes definidas não correspondem às classes do modelo!")
else:
    print("Classes verificadas com sucesso!")

# Carregar a imagem
image_path = r"C:\Users\danil\Downloads\Atualizado\results\frame0000.jpg"  # Substitua pelo caminho da sua imagem
image = cv2.imread(image_path)
if image is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")
    exit()

# Executar a predição com thresholds reduzidos
results = model.predict(image, conf=0.1, iou=0.45, verbose=True)

# Verificar resultados
for result in results:
    # Verificar se há caixas delimitadoras
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"Caixas detectadas: {len(result.boxes)}")
    else:
        print("Nenhuma caixa detectada")

    # Verificar máscaras
    if result.masks is not None:
        print(f"Máscaras detectadas: {len(result.masks)}")
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        # Iterar sobre as detecções
        for i in range(len(masks)):
            mask = masks[i]
            class_id = int(classes[i])
            score = scores[i]
            label = f"{class_names[class_id]}: {score:.2f}"
            print(f"Detecção {i+1}: Classe={class_names[class_id]}, Confiança={score:.2f}")

            # Redimensionar a máscara
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8) * 255

            # Sobrepor a máscara
            overlay = image.copy()
            color = np.random.randint(0, 255, size=3).tolist()
            overlay[mask == 255] = color
            alpha = 0.4
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Desenhar caixa delimitadora
            box = boxes[i].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Adicionar rótulo
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (box[0], box[1] - label_h - 10), (box[0] + label_w, box[1]), color, -1)
            cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        print("Nenhuma máscara detectada")

# Salvar e exibir a imagem resultante
output_path = "output_segmented_image.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Imagem processada salva em: {output_path}")