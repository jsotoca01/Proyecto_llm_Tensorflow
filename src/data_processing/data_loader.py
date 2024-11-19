from datasets import load_dataset
import os
from src.config_loader import load_config

config = load_config()  # Carga la configuración del proyecto

def load_imdb_data():
    """
    Carga el dataset de IMDb desde Hugging Face y guarda las reseñas en un archivo de texto.
    """
    """
    Descarga y carga el dataset de IMDb desde Hugging Face.
    Extrae las reseñas y las guarda en un archivo de texto.
    """
    if not config:
        print("Error: Configuración no cargada.")
        return

    # Cargar el dataset de IMDb desde Hugging Face
    dataset = load_dataset("imdb")
    print("Dataset cargado con éxito. Columnas del dataset:", dataset["train"].column_names)
    
    # Extraer las reseñas de texto del conjunto de entrenamiento
    text_data = dataset["train"]["text"]
    labels = dataset["train"]["label"]
    print("Ejemplo de reseña:", text_data[0], labels[0])

    # Usar la ruta desde el archivo de configuración
    raw_data_path = config["data"]["raw_data_path"]
    os.makedirs(raw_data_path, exist_ok=True)
    
    with open(os.path.join(raw_data_path, "imdb_reviews.txt"), "w", encoding="utf-8") as file:
        for review in text_data:
            file.write(review + "\n")
    
    print("Reseñas de IMDb guardadas en imdb_reviews.txt")
    
    with open(os.path.join(raw_data_path, "imdb_labels.txt"), "w", encoding="utf-8") as file_labels:
        for label in labels:
            file_labels.write(str(label) + "\n")

if __name__ == "__main__":
    load_imdb_data()
