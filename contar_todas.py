from ultralytics import YOLO
import os

# Cargar tu modelo entrenado
model = YOLO("runs/detect/train2/weights/best.pt")

# Carpeta con imágenes
carpeta = "images/train"

print("Conteo de personas por imagen:\n")

for img in os.listdir(carpeta):
    if img.lower().endswith((".jpg", ".png")):
        ruta = os.path.join(carpeta, img)
        results = model(ruta, conf=0.4, verbose=False)
        conteo = len(results[0].boxes)
        print(f"{img}: {conteo} personas")