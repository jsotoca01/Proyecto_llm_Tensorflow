import tensorflow as tf
import numpy as np
import os
from src.config_loader import load_config  # Importar la función para cargar la configuración desde config.yaml
from src.model.model import TransformerModel  # Importar el modelo Transformer personalizado
from src.model.train import get_optimizer, train_step, train_accuracy  # Importar el optimizador, la función de entrenamiento y la métrica de precisión

# Cargar la configuración desde config.yaml, que contiene parámetros del modelo y de entrenamiento
config = load_config()

def main():
    """
    Ejecuta el flujo principal de entrenamiento:
    - Carga los datos y parámetros del modelo.
    - Inicializa el modelo y el optimizador.
    - Configura checkpoints para guardar el progreso de entrenamiento.
    - Entrena el modelo durante las épocas especificadas y guarda checkpoints al final de cada época.
    """
    print("Iniciando entrenamiento...")

    # Paso 1: Cargar datos procesados desde el archivo .npy en data/processed
    processed_data_path = config["data"]["processed_data_path"]  # Obtener la ruta de los datos procesados desde config.yaml
    data = np.load(processed_data_path)  # Cargar el array numpy con las secuencias de texto tokenizadas
    x_train = data  # Asignar los datos de entrenamiento
    y_train = np.random.randint(2, size=(x_train.shape[0],))  # Crear etiquetas aleatorias como ejemplo (en un proyecto real usarías etiquetas reales)

    # Paso 2: Inicializar el modelo, optimizador y configuración de checkpoints
    model = TransformerModel(
        vocab_size=config["model"]["vocab_size"],  # Tamaño del vocabulario
        d_model=config["model"]["d_model"],  # Dimensión de los embeddings
        num_heads=config["model"]["num_heads"],  # Número de cabezas en la capa de atención
        dff=config["model"]["dff"],  # Dimensión de la red feedforward
        input_length=config["model"]["input_length"],  # Longitud máxima de entrada
        num_layers=config["model"]["num_layers"]  # Número de capas del encoder
    )

    optimizer = get_optimizer()  # Obtener el optimizador configurado (por ejemplo, Adam)

    # Configuración de checkpoints para guardar el estado del modelo y del optimizador
    checkpoint_dir = config["checkpoints"]["checkpoint_dir"]  # Ruta para guardar los checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)  # Crear el directorio de checkpoints si no existe

    # Definir una variable de contador de épocas para rastrear el progreso
    epoch_counter = tf.Variable(1, trainable=False, dtype=tf.int64)  # Variable que cuenta la época actual para poder reanudar el entrenamiento

    # Crear un checkpoint que incluye el modelo, el optimizador y el contador de épocas
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=epoch_counter)

    # Configurar el checkpoint manager para manejar los checkpoints y especificar cuántos guardar (max_to_keep=3)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Restaurar el último checkpoint guardado si existe, para reanudar el entrenamiento desde la última época
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Checkpoint restaurado desde {checkpoint_manager.latest_checkpoint}. Reanudando desde la época {epoch_counter.numpy()}")

    # Paso 3: Configurar el ciclo de entrenamiento, incluyendo el número de épocas y el tamaño de los batches
    epochs = config["training"]["epochs"]  # Número total de épocas desde config.yaml
    batch_size = config["training"]["batch_size"]  # Tamaño del batch desde config.yaml

    # Ciclo principal de entrenamiento: se ejecuta desde la última época guardada hasta el total de épocas
    for epoch in range(epoch_counter.numpy(), epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Reiniciar la métrica de precisión al inicio de cada época para que no acumule valores de épocas anteriores
        train_accuracy.reset_states()

        # Dividir los datos en batches y realizar el entrenamiento para cada uno
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]  # Extraer el batch actual de entradas
            y_batch = y_train[i:i + batch_size]  # Extraer el batch actual de etiquetas correspondientes
            loss = train_step(model, optimizer, x_batch, y_batch)  # Realizar un paso de entrenamiento y calcular la pérdida del batch actual

            # Imprimir la pérdida y precisión en el batch actual para monitorear el progreso
            print(f"Batch {i // batch_size + 1}, Loss: {loss.numpy()}, Accuracy: {train_accuracy.result().numpy()}")

        # Guardar un checkpoint al final de cada época para poder reanudar en caso de interrupción
        epoch_counter.assign_add(1)  # Incrementar el contador de épocas
        checkpoint_manager.save()  # Guardar el estado actual del modelo, optimizador y contador de épocas
        print(f"Checkpoint guardado al final de la época {epoch}")

    print("Entrenamiento completado.")

if __name__ == "__main__":
    main()
