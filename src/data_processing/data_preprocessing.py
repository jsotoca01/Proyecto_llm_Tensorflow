from src.data_processing.data_loader import load_imdb_data
from src.data_processing.preprocessor import DataPreprocessor

def main():
    """
    Orquesta el flujo de procesamiento de datos:
    1. Carga los datos en bruto.
    2. Preprocesa los datos y los guarda.
    """
    print("Iniciando el procesamiento de datos...")

    # Paso 1: Cargar datos en bruto
    load_imdb_data()

    # Paso 2: Preprocesar datos
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()

    print("Procesamiento de datos completado.")

if __name__ == "__main__":
    main()

