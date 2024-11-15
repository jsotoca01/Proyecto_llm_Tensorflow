import yaml
import os

def load_config():
    """
    Carga la configuración desde el archivo config.yaml.

    Retorna:
    - config: Un diccionario con las configuraciones.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            print("Configuración cargada correctamente.")
            return config
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de configuración en {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error al analizar el archivo YAML: {e}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    config = load_config()
    if config:
        print(config)
