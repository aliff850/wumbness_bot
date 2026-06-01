"""
Convert a Keras (.keras) model to ONNX (.onnx) format.

Usage:
    python convert_to_onnx.py

This script requires `tensorflow` and `tf2onnx` to be installed:

    pip install tensorflow tf2onnx

After conversion you can uninstall both and use only `onnxruntime` at runtime.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Patch Keras deserialization so older models that contain
# 'quantization_config' fields can still be loaded.
# ---------------------------------------------------------------------------
import keras
import keras.src.saving.serialization_lib


def _clean_config(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for key, value in list(config.items()):
            _clean_config(value)
    elif isinstance(config, list):
        for item in config:
            _clean_config(item)


_orig_deserialize = keras.saving.deserialize_keras_object


def _patched_deserialize(config, custom_objects=None, safe_mode=True, **kwargs):
    _clean_config(config)
    return _orig_deserialize(
        config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs
    )


keras.saving.deserialize_keras_object = _patched_deserialize
keras.src.saving.serialization_lib.deserialize_keras_object = _patched_deserialize

# ---------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import tf2onnx

# --- Configuration --------------------------------------------------------
KERAS_MODEL_PATH = "optimized_nostop_bidirectional_lstm_model_10epoch.keras"
ONNX_OUTPUT_PATH = "optimized_nostop_bidirectional_lstm_model_10epoch.onnx"
# --------------------------------------------------------------------------


def convert():
    # Resolve paths relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    keras_path = os.path.join(base_dir, KERAS_MODEL_PATH)
    onnx_path = os.path.join(base_dir, ONNX_OUTPUT_PATH)

    if not os.path.isfile(keras_path):
        print(f"❌ Keras model not found at: {keras_path}")
        sys.exit(1)

    print(f"Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path)
    model.summary()

    # Define the input signature so tf2onnx knows the expected shape and dtype.
    # The model expects (batch_size, max_len) where max_len = 150.
    input_signature = [
        tf.TensorSpec(shape=(None, 150), dtype=tf.int32, name="input"),
    ]

    print("\nConverting to ONNX via function tracking...")
    
    # 1. Force compilation into a traceable function
    @tf.function(input_signature=input_signature)
    def model_func(input_tensor):
        return model(input_tensor)

    # 2. Convert using from_function (Pass model_func directly!)
    model_proto, _ = tf2onnx.convert.from_function(
        model_func, 
        input_signature=input_signature,
        output_path=onnx_path,
        opset=13,  # opset 13 has excellent, stable LSTM kernel fusion support
    )

    original_size_mb = os.path.getsize(keras_path) / (1024 * 1024)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    print(f"\n✅ Conversion complete!")
    print(f"   Original Keras model : {original_size_mb:.2f} MB  ({keras_path})")
    print(f"   ONNX model           : {onnx_size_mb:.2f} MB  ({onnx_path})")
    print(f"   Size reduction       : {(1 - onnx_size_mb / original_size_mb) * 100:.1f}%")

if __name__ == "__main__":
    convert()
