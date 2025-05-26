from ultralytics import YOLO
import cv2

# Cores fixas por classe
CLASS_COLORS = {
    0: (0, 255, 255),  # Amarelo (classe 0)
    1: (0, 0, 255),    # Vermelho (classe 1)
}

# Nomes de classe (ajuste conforme necessário)
CLASS_NAMES = {
    0: 'Passando',
    1: 'Maduros',
    2: 'Semi-Maduro',
    3: 'Verde',
}

# Cor do texto (laranja escuro em BGR)
TEXT_COLOR = (255, 0, 50)

def main():
    model = YOLO(r'C:\Users\danil\Downloads\Atualizado\modelos\CoffeCherryYoloL\weights\best.pt')

    results = model.track(
        source='cafesplantaszapb.mp4',
        conf=0.3,
        stream=True,
        tracker='botsort.yaml'
    )

    for result in results:
        frame = result.orig_img.copy()
        counts = {0: 0, 1: 0}

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box[:4]
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if cls in counts:
                    counts[cls] += 1
                else:
                    counts[cls] = 1

        # Mostrar a contagem no canto superior esquerdo da imagem
        y_offset = 30
        for cls_id, count in counts.items():
            text = f"{CLASS_NAMES.get(cls_id, f'Classe {cls_id}')}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            y_offset += 30

        cv2.imshow('Detecção com Contagem', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
