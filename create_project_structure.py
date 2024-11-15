import os

# Define la estructura del proyecto
project_structure = {
    "config": ["config.yaml"],
    "data/raw": [],
    "data/processed": [],
    "docs": ["usage.md"],
    "src": ["__init__.py", "config_loader.py", "train_main.py", "utils.py"],
    "src/data_processing": ["__init__.py", "data_loader.py", "preprocessor.py", "data_preprocessing.py"],
    "src/model": ["__init__.py", "train.py", "model.py"],
    "tests": ["__init__.py", "test_data_processing.py", "test_model.py"],
}

# Contenidos base para archivos clave en la raíz del proyecto
root_files = {
    "README.md": """# ProyectoLLM

Este proyecto implementa un modelo de lenguaje grande (LLM) utilizando Python y TensorFlow. Incluye el procesamiento de datos, la construcción del modelo y el entrenamiento.

## Requisitos

- Python 3.x
- TensorFlow
- Otros paquetes listados en `requirements.txt`

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/ProyectoLLM.git
   cd ProyectoLLM
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta el procesamiento de datos o el entrenamiento desde `main.py`:
```bash
python main.py data_processing
python main.py train_main
```

## Estructura del Proyecto

- `config/`: Configuración del proyecto.
- `data/`: Almacena los datos sin procesar y procesados.
- `src/`: Código fuente principal.
- `tests/`: Pruebas unitarias.
- `docs/`: Documentación adicional.

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.
""",

    "LICENSE": """MIT License

Copyright (c) 2024 Tu Nombre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
""",

    "requirements.txt": """tensorflow
datasets
numpy
PyYAML
""",

    "setup.py": """from setuptools import setup, find_packages

setup(
    name="ProyectoLLM",
    version="0.1.0",
    author="Tu Nombre",
    description="Implementación de un modelo de lenguaje grande (LLM) con TensorFlow.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow",
        "datasets",
        "numpy",
        "PyYAML"
    ],
)
""",

    "config/config.yaml": """data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/processed_imdb_reviews.npy"

model:
  vocab_size: 10000
  d_model: 128
  num_heads: 8
  dff: 512
  input_length: 512
  num_layers: 4

training:
  batch_size: 32
  buffer_size: 10000
  epochs: 10
  warmup_steps: 4000
""",

    ".gitignore": """# Archivos y carpetas a ignorar por Git
__pycache__/
*.py[cod]
data/processed/
data/raw/
env/
.venv/
.vscode/
.idea/
"""
}

# Función para crear la estructura de carpetas y archivos
def create_project_structure(base_path, structure, root_files):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)  # Crea la carpeta si no existe
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):  # Evita sobrescribir archivos existentes
                with open(file_path, 'w') as f:
                    # Añade un comentario base en los archivos de código
                    if file.endswith(".py"):
                        f.write(f"# {file}\n")
                    else:
                        f.write("")  # Archivos vacíos para otras extensiones

    # Crear archivos raíz con contenido específico
    for file, content in root_files.items():
        file_path = os.path.join(base_path, file)
        with open(file_path, 'w') as f:
            f.write(content)

# Define la ruta base como la carpeta actual
base_path = os.getcwd()
create_project_structure(base_path, project_structure, root_files)

print("Estructura del proyecto creada con éxito.")
