import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import keras.src.saving.serialization_lib

# Patch Keras deserialization to handle models saved in older Keras versions
# that contain 'quantization_config' fields which the current Keras version rejects.
def _clean_config(config):
    if isinstance(config, dict):
        config.pop('quantization_config', None)
        for key, value in list(config.items()):
            _clean_config(value)
    elif isinstance(config, list):
        for item in config:
            _clean_config(item)

_orig_deserialize = keras.saving.deserialize_keras_object
def _patched_deserialize(config, custom_objects=None, safe_mode=True, **kwargs):
    _clean_config(config)
    return _orig_deserialize(config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs)

keras.saving.deserialize_keras_object = _patched_deserialize
keras.src.saving.serialization_lib.deserialize_keras_object = _patched_deserialize

class CyberbullyingPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        # Resolve paths dynamically relative to this file's folder if paths are relative
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(base_dir, model_path))
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.abspath(os.path.join(base_dir, tokenizer_path))
            
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.max_len = 150

    def predict(self, text: str) -> dict:
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict probabilities
        probs = self.model.predict(padded)[0]
        
        return {label: float(prob) for label, prob in zip(self.labels, probs)}

