import tensorflow as tf
import numpy as np
from src.config_loader import load_config

config = load_config()  # Cargar configuración desde config.yaml

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Define una capa de codificación del Transformer.

        Parámetros:
        - d_model: Dimensión del modelo (embedding).
        - num_heads: Número de cabezas en la capa de atención.
        - dff: Dimensión de la capa feedforward.
        - rate: Tasa de dropout para la regularización.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),  # Capa feedforward con ReLU
            tf.keras.layers.Dense(d_model)  # Capa de salida para mantener la dimensión original
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        # Capa de atención multi-cabeza
        attn_output = self.mha(x, x, x)  # Auto-atención sobre x
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual + Normalización

        # Capa feedforward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual + Normalización

        return out2


class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, input_length, num_layers):
        """
        Inicializa el modelo Transformer.

        Parámetros:
        - vocab_size: Tamaño del vocabulario.
        - d_model: Dimensión de los embeddings.
        - num_heads: Número de cabezas en la capa de atención.
        - dff: Dimensión de la capa feedforward.
        - input_length: Longitud de las secuencias de entrada.
        - num_layers: Número de capas de encoder.
        """
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.get_positional_encoding(input_length, d_model)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]
        self.final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def get_positional_encoding(self, max_seq_len, d_model):
        """
        Genera el encoding posicional para capturar la posición de las palabras en la secuencia.

        Parámetros:
        - max_seq_len: Longitud máxima de la secuencia de entrada.
        - d_model: Dimensión del embedding.

        Retorna:
        - Tensor con el encoding posicional de forma [max_seq_len, d_model].
        """
        pos_encoding = np.zeros((max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        """
        Llamada al modelo para realizar el forward pass.

        Parámetros:
        - x: Tensor de entrada.
        - training: Indica si el modelo está en modo de entrenamiento.

        Retorna:
        - Salida del modelo después de aplicar las capas de atención y feedforward.
        """
        seq_len = tf.shape(x)[1]  # Longitud de la secuencia de entrada
        x = self.embedding(x)  # Convertir las entradas en embeddings
        x += self.pos_encoding[:seq_len, :]  # Ajustar el pos_encoding a la longitud de la secuencia

        # Pasar por cada capa de codificación
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # Usar la salida de la primera posición para la clasificación binaria
        return self.final_layer(x[:, 0, :])

if __name__ == "__main__":
    # Cargar configuraciones y parámetros desde config.yaml
    sample_input = tf.constant([[1, 2, 3, 4, 5]])  # Entrada de ejemplo
    model = TransformerModel(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        dff=config["model"]["dff"],
        input_length=config["model"]["input_length"],
        num_layers=config["model"]["num_layers"]
    )
    output = model(sample_input)
    print("Salida del modelo en prueba:", output)

