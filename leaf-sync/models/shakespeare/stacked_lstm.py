import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        # Placeholders
        features = tf.placeholder(tf.int32, [None, self.seq_len], name="features")
        labels   = tf.placeholder(tf.int32, [None, self.num_classes], name="labels")

        # Tabla de embeddings (vocab_size = num_classes)
        embedding = tf.get_variable("embedding", [self.num_classes, 8])

        # --- AIRBAG: clip de índices para evitar valores fuera de rango (-1, etc.) ---
        # Si algún carácter fuera del vocabulario aparece, lo mapeamos al rango válido [0, vocab_size-1].
        vocab_static = embedding.get_shape().as_list()[0]  # normalmente == self.num_classes
        if vocab_static is None:
            vocab_size = tf.shape(embedding)[0]  # dinámico
        else:
            vocab_size = vocab_static            # estático (int)

        features_i32 = tf.cast(features, tf.int32)
        if isinstance(vocab_size, int):
            features_safe = tf.clip_by_value(features_i32, 0, vocab_size - 1)
        else:
            features_safe = tf.minimum(tf.maximum(features_i32, 0), vocab_size - 1)

        x = tf.nn.embedding_lookup(embedding, features_safe)

        # Modelo LSTM apilado
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)]
        )
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)

        # Capa densa final
        pred = tf.layers.dense(inputs=outputs[:, -1, :], units=self.num_classes)

        # Pérdida (casteamos labels a float32 para ser estrictos con TF1)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=pred,
                labels=tf.cast(labels, tf.float32)
            )
        )

        # Optimizador
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        # Métrica de accuracy (conteo de aciertos)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        # Palabras -> índices; garantizamos dtype int32 para el placeholder
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch, dtype=np.int32)
        return x_batch

    def process_y(self, raw_y_batch):
        # One-hot de caracteres; dejamos int32 (se castea a float32 en la pérdida)
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        y_batch = np.array(y_batch, dtype=np.int32)
        return y_batch
