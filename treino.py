from ultralytics import YOLO

if __name__ == '__main__':
    # Carregar o modelo pré-treinado YOLO11m
    model = YOLO("yolo11m.pt")

    # Treinar o modelo
    model.train(
    data=r"C:\Users\danil\Downloads\Atualizado\datasets\Coffee-Cherry-1\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="CoffeCherryFruitsYoloMCustom",
    save_period=10,
    patience= 20 
)