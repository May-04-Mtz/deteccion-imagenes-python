from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train2/weights/best.pt")

entrada = "images/train"
salida = "imagenes_resultado"
os.makedirs(salida, exist_ok=True)

for img in os.listdir(entrada):
    if img.lower().endswith((".jpg", ".png")):
        ruta = os.path.join(entrada, img)
        image = cv2.imread(ruta)

        results = model(ruta, conf=0.4, verbose=False)
        conteo = len(results[0].boxes)

        # Dibujar detecciones
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Texto del conteo
        cv2.putText(
            image,
            f"Personas: {conteo}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imwrite(os.path.join(salida, img), image)

print("✅ Imágenes guardadas en 'imagenes_resultado'")
