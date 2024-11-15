from setuptools import setup, find_packages

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
