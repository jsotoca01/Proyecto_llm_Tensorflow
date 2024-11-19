import os
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config_loader import load_config

# Cargar configuración desde config.yaml
config = load_config()

class DataPreprocessor:
    def __init__(self):
        """
        Inicializa el preprocesador de datos.
        """
        self.tokenizer = Tokenizer(num_words=config["model"]["vocab_size"], oov_token="<OOV>")
        self.max_length = config["model"]["input_length"]

    @staticmethod
    def clean_text(text):
        """
        Limpia el texto eliminando etiquetas HTML, caracteres no deseados y múltiples espacios.

        Parámetros:
        - text: cadena de texto a limpiar.

        Retorna:
        - Texto limpio en minúsculas.
        """
        # Eliminar etiquetas HTML como <br />
        text = re.sub(r"<.*?>", " ", text)
        # Eliminar caracteres no alfanuméricos (excepto espacios)
        #text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        # Convertir múltiples espacios en uno solo
        text = re.sub(r"\s+", " ", text)
        # Convertir el texto a minúsculas y eliminar espacios extra
        #return text.lower().strip()
        return text.strip()

    def preprocess_data(self):
        """
        Preprocesa el texto de las reseñas: limpieza, tokenización, padding y guardado en disco.
        """
        # Ruta al archivo con las reseñas sin procesar
        raw_data_path = config["data"]["raw_data_path"]

        # Leer reseñas desde el archivo
        with open(os.path.join(raw_data_path, "imdb_reviews.txt"), "r", encoding="utf-8") as file:
            reviews = file.readlines()

        # Leer etiquetas desde el archivo
        with open(os.path.join(raw_data_path, "imdb_labels.txt"), "r", encoding="utf-8") as file_labels:
            labels = [int(label.strip()) for label in file_labels.readlines()]

        # Limpiar reseñas antes de la tokenización
        print("Limpiando texto de las reseñas...")
        reviews = [self.clean_text(review) for review in reviews]
        # Mostrar la primera reseña limpia
        print("Primera reseña limpia:", reviews[0])

        # Tokenización y padding
        print("Tokenizando texto...")
        self.tokenizer.fit_on_texts(reviews)
        sequences = self.tokenizer.texts_to_sequences(reviews)

        print("Aplicando padding a las secuencias...")
        padded_sequences = pad_sequences(
            sequences, maxlen=self.max_length, padding="post", truncating="post"
        )

        # Crear la carpeta de datos procesados si no existe
        processed_data_path = config["data"]["processed_data_path"]
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

        # Guardar datos procesados en un archivo .npy
        print(f"Guardando los datos procesados en: {processed_data_path}")
        data_to_save = {
            "sequences": padded_sequences,  # Secuencias tokenizadas y rellenadas
            "labels": labels  # Etiquetas correspondientes
        }
        np.save(processed_data_path, data_to_save)
        print(f"Datos procesados guardados exitosamente en {processed_data_path}")
