from ultralytics import YOLO
import os
import csv

# Cargar modelo entrenado
model = YOLO("runs/detect/train2/weights/best.pt")

# Carpeta de imágenes
carpeta = "images/train"

# Archivo CSV
archivo_csv = "conteo_personas.csv"

# Obtener y ordenar imágenes
imagenes = sorted([
    img for img in os.listdir(carpeta)
    if img.lower().endswith((".jpg", ".png"))
])

total_personas = 0

with open(archivo_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Encabezados
    writer.writerow(["No.", "Imagen", "Personas detectadas"])

    print("\nConteo por imagen:\n")

    for i, img in enumerate(imagenes, start=1):
        ruta = os.path.join(carpeta, img)
        results = model(ruta, conf=0.4, verbose=False)
        conteo = len(results[0].boxes)

        total_personas += conteo

        writer.writerow([i, img, conteo])
        print(f"{i}. {img}: {conteo} personas")

    # Cálculos finales
    total_imagenes = len(imagenes)
    promedio = total_personas / total_imagenes if total_imagenes > 0 else 0

    # Espacio y resumen
    writer.writerow([])
    writer.writerow(["TOTAL", "", total_personas])
    writer.writerow(["PROMEDIO POR AUTOBÚS", "", round(promedio, 2)])

print("\n📈 Total de personas:", total_personas)
print("🚌 Promedio por autobús:", round(promedio, 2))
print("\n✅ Archivo 'conteo_personas.csv' actualizado")