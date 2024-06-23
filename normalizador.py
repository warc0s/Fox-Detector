import os
import pandas as pd
import imghdr
from PIL import Image

# Rutas de las carpetas originales
path_no_zorros = 'no_zorros'
path_si_zorros = 'si_zorros'

# Crear una carpeta para el nuevo dataset si no existe
new_dataset_path = 'images_dataset'
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)
    print(f"Creada nueva carpeta para el dataset: {new_dataset_path}")

# Función para redimensionar y guardar imágenes
def process_and_save_image(image_path, output_path, file_names):
    with Image.open(image_path) as img:
        # Convertir a "RGB" si la imagen está en modo "P"
        if img.mode == 'P':
            img = img.convert('RGB')

        # Redimensionar a 300x300 sin mantener el aspecto
        img_resized = img.resize((300, 300), Image.Resampling.LANCZOS)

        # Verificar si el nombre de archivo ya existe y modificarlo si es necesario
        base_name, extension = os.path.splitext(output_path)
        counter = 1
        new_output_path = output_path
        while os.path.exists(new_output_path):
            new_output_path = f"{base_name}_{counter}{extension}"
            counter += 1

        img_resized.save(new_output_path, quality=95)  # Guardar con alta calidad
        file_names.append(os.path.basename(new_output_path))

# Listas para almacenar la información del CSV
file_names = []
labels = []

# Procesar imágenes de no zorros
print("Procesando imágenes de no zorros...")
for filename in os.listdir(path_no_zorros):
    file_path = os.path.join(path_no_zorros, filename)
    if imghdr.what(file_path):  # Verifica si el archivo es una imagen
        new_file_path = os.path.join(new_dataset_path, filename)
        process_and_save_image(file_path, new_file_path, file_names)
        labels.append(0)  # 0 para no zorros

# Procesar imágenes de sí zorros
print("Procesando imágenes de sí zorros...")
for filename in os.listdir(path_si_zorros):
    file_path = os.path.join(path_si_zorros, filename)
    if imghdr.what(file_path):  # Verifica si el archivo es una imagen
        new_file_path = os.path.join(new_dataset_path, filename)
        process_and_save_image(file_path, new_file_path, file_names)
        labels.append(1)  # 1 para sí zorros

# Crear DataFrame y guardar como CSV
print("Creando archivo CSV...")
dataset = pd.DataFrame({'filename': file_names, 'label': labels})
csv_path = 'dataset.csv'
dataset.to_csv(csv_path, index=False)
print(f"Archivo CSV creado con éxito: {csv_path}")