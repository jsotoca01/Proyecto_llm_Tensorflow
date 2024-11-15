import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config_loader import load_config

config = load_config()  # Carga la configuración del proyecto

class DataPreprocessor:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=config["model"]["vocab_size"], oov_token="<OOV>")
        self.max_length = config["model"]["input_length"]
    
    def preprocess_data(self):
        """
        Preprocesa el texto de las reseñas: tokenización, padding y guardado.
        """
        # Cargar el archivo de texto con las reseñas
        raw_data_path = config["data"]["raw_data_path"]
        with open(os.path.join(raw_data_path, "imdb_reviews.txt"), "r", encoding="utf-8") as file:
            reviews = file.readlines()

        # Tokenización y padding
        self.tokenizer.fit_on_texts(reviews)
        sequences = self.tokenizer.texts_to_sequences(reviews)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")

         # Código existente...
    
        # Crear la carpeta de datos procesados si no existe
        processed_data_path = config["data"]["processed_data_path"]
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

        # Verificar y mostrar la ruta antes de guardar
        print(f"Intentando guardar los datos procesados en: {processed_data_path}")
        
        # Guardar los datos procesados en un archivo .npy
        processed_data_path = config["data"]["processed_data_path"]
        np.save(processed_data_path, padded_sequences)
        print(f"Datos procesados guardados en {processed_data_path}")

