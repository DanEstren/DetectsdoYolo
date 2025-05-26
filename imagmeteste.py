from ultralytics import YOLO
import cv2

imageUrl = 'IMG_6583-1.jpg'

# Load a model
model = YOLO(r'C:\Users\danil\Downloads\Atualizado\modelos\CafesSegmentacaoNanoTeste502\weights\best.pt')
model.conf = 0.5

# Predict with the model
results = model(imageUrl)  # predict on an image

results_image = results[0].plot()
cv2.imshow('Result', results_image)
cv2.waitKey(0)
cv2.destroyAllWindows()