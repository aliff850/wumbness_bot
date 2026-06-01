import os
import sys
import json
import re
import numpy as np

# =========================================================================
# ONNX Runtime Implementation (active)
# =========================================================================
import onnxruntime as ort


def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    Pure NumPy replacement for keras pad_sequences.
    """
    result = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if truncating == 'post':
            trunc = seq[:maxlen]
        else:
            trunc = seq[-maxlen:]
        if padding == 'post':
            result[i, :len(trunc)] = trunc
        else:
            result[i, -len(trunc):] = trunc
    return result

class CyberbullyingPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(base_dir, model_path))
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.abspath(os.path.join(base_dir, tokenizer_path))

        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer dictionary from: {tokenizer_path}")

        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Load the raw dictionary instead of the Keras object
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            self.word_index = json.load(f)
            
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.max_len = 150

    def texts_to_sequences(self, text: str) -> list:
        """
        Pure Python implementation of Keras texts_to_sequences
        """
        # Clean text (mimicking Keras default filters)
        text = text.lower()
        text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', '', text)
        words = text.split()
        
        # Map words to integers based on the JSON dictionary
        sequence = []
        for word in words:
            if word in self.word_index:
                sequence.append(self.word_index[word])
        return [sequence] # Return as a list of sequences to match batch format

    def predict(self, text: str) -> dict:
        # Use our custom pure-Python sequence converter
        seq = self.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')

        # Run inference via ONNX Runtime
        probs = self.session.run(None, {self.input_name: padded})[0][0]

        return {label: float(prob) for label, prob in zip(self.labels, probs)}

# =========================================================================
# Old TFLite Implementation (commented out)
# =========================================================================
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import tflite_runtime.interpreter as tflite
#
# class CyberbullyingPredictor:
#     def __init__(self, model_path: str, tokenizer_path: str):
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
#         if not os.path.isabs(model_path):
#             model_path = os.path.abspath(os.path.join(base_dir, model_path))
#         if not os.path.isabs(tokenizer_path):
#             tokenizer_path = os.path.abspath(os.path.join(base_dir, tokenizer_path))
#
#         print(f"Loading model from: {model_path}")
#         print(f"Loading tokenizer from: {tokenizer_path}")
#
#         self.interpreter = tflite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#
#         self.tokenizer = joblib.load(tokenizer_path)
#         self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
#         self.max_len = 150
#
#     def predict(self, text: str) -> dict:
#         seq = self.tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
#
#         input_dtype = self.input_details[0]['dtype']
#         padded = padded.astype(input_dtype)
#
#         self.interpreter.set_tensor(self.input_details[0]['index'], padded)
#         self.interpreter.invoke()
#
#         probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
#
#         return {label: float(prob) for label, prob in zip(self.labels, probs)}


# =========================================================================
# Old TensorFlow Implementation (commented out)
# =========================================================================
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import keras
# import keras.src.saving.serialization_lib
#
# # Patch Keras deserialization to handle models saved in older Keras versions
# # that contain 'quantization_config' fields which the current Keras version rejects.
# def _clean_config(config):
#     if isinstance(config, dict):
#         config.pop('quantization_config', None)
#         for key, value in list(config.items()):
#             _clean_config(value)
#     elif isinstance(config, list):
#         for item in config:
#             _clean_config(item)
#
# _orig_deserialize = keras.saving.deserialize_keras_object
# def _patched_deserialize(config, custom_objects=None, safe_mode=True, **kwargs):
#     _clean_config(config)
#     return _orig_deserialize(config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs)
#
# keras.saving.deserialize_keras_object = _patched_deserialize
# keras.src.saving.serialization_lib.deserialize_keras_object = _patched_deserialize
#
# class CyberbullyingPredictor:
#     def __init__(self, model_path: str, tokenizer_path: str):
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
#         if not os.path.isabs(model_path):
#             model_path = os.path.abspath(os.path.join(base_dir, model_path))
#         if not os.path.isabs(tokenizer_path):
#             tokenizer_path = os.path.abspath(os.path.join(base_dir, tokenizer_path))
#
#         print(f"Loading model from: {model_path}")
#         print(f"Loading tokenizer from: {tokenizer_path}")
#
#         self.model = tf.keras.models.load_model(model_path)
#         self.tokenizer = joblib.load(tokenizer_path)
#         self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
#         self.max_len = 150
#
#     def predict(self, text: str) -> dict:
#         seq = self.tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
#
#         # Predict probabilities
#         probs = self.model.predict(padded)[0]
#
#         return {label: float(prob) for label, prob in zip(self.labels, probs)}
