import tensorflow as tf
from .model import TransformerModel  # Importación directa para evitar dependencias cíclicas
from src.config_loader import load_config

config = load_config()

def get_model():
    """
    Inicializa el modelo Transformer con los parámetros desde config.yaml.
    """
    return TransformerModel(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        dff=config["model"]["dff"],
        input_length=config["model"]["input_length"],
        num_layers=config["model"]["num_layers"]
    )

def get_optimizer():
    """
    Define el optimizador con un programador de tasa de aprendizaje personalizado.
    """
    return tf.keras.optimizers.Adam()

# Nueva instancia de la métrica de precisión
train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

@tf.function
def train_step(model, optimizer, x, y):
    """
    Realiza un paso de entrenamiento: calcula la pérdida, ajusta los pesos y actualiza la precisión.
    """
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        # Ajustar la forma de y para que coincida con predictions
        y = tf.reshape(y, (-1, 1))  # Cambia y a la forma (batch_size, 1)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Actualizar la métrica de precisión con las predicciones del batch actual
    train_accuracy.update_state(y, predictions)

    return loss

