import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path, modelname="model32"):
        self.model = tf.keras.models.load_model(os.path.join(path, modelname))
        #'training_1_SK_BRANN_CL_15_21'
    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out

