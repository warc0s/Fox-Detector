import os
import tensorflow as tf
import imghdr
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from shutil import move

# Cargar el modelo
model = tf.keras.models.load_model('fox_detector_model.keras')

# Directorios
base_dir = './'  # Ajusta según la ubicación de tus carpetas
source_dir = os.path.join(base_dir, 'todas_imagenes')
si_es_zorro_dir = os.path.join(base_dir, 'si_es_zorro')
no_es_zorro_dir = os.path.join(base_dir, 'no_es_zorro')

# Verificar si existe la carpeta "todas_imagenes"
if not os.path.isdir(source_dir):
    print(f"La carpeta '{source_dir}' no existe. Por favor, crea la carpeta con imágenes y vuelve a ejecutar el script.")
    exit()

# Crear las carpetas "si_es_zorro" y "no_es_zorro" si no existen
os.makedirs(si_es_zorro_dir, exist_ok=True)
os.makedirs(no_es_zorro_dir, exist_ok=True)

# Procesar cada archivo en la carpeta "todas_imagenes"
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    # Verificar si el archivo es una imagen
    if imghdr.what(file_path):
        # Cargar y preparar la imagen
        img = load_img(file_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalizar los píxeles
        img_array = tf.expand_dims(img_array, 0)  # Crear un lote que contenga una sola imagen

        # Predecir la clase
        prediction = model.predict(img_array)
        class_idx = int(prediction[0][0] > 0.5)  # Interpretar la predicción como 1 o 0

        # Mover la imagen al directorio correspondiente
        if class_idx == 1:
            move(file_path, os.path.join(si_es_zorro_dir, filename))
        else:
            move(file_path, os.path.join(no_es_zorro_dir, filename))

print("Clasificación y organización de imágenes completada.")