"""
Convert a Keras (.keras) model to TensorFlow Lite (.tflite) format.

Usage:
    python convert_to_tflite.py

This script requires the full `tensorflow` package to be installed
(not just tflite-runtime), because the converter lives in TensorFlow.

    pip install tensorflow

After conversion you can uninstall tensorflow and use only tflite-runtime.
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

# ---------------------------------------------------------------------------

import tensorflow as tf

# --- Configuration --------------------------------------------------------
KERAS_MODEL_PATH = "optimized_nostop_bidirectional_lstm_model_10epoch.keras"
TFLITE_OUTPUT_PATH = "optimized_nostop_bidirectional_lstm_model_10epoch.tflite"
# --------------------------------------------------------------------------

def convert():
    # Resolve paths relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    keras_path = os.path.join(base_dir, KERAS_MODEL_PATH)
    tflite_path = os.path.join(base_dir, TFLITE_OUTPUT_PATH)

    if not os.path.isfile(keras_path):
        print(f"❌ Keras model not found at: {keras_path}")
        sys.exit(1)

    print(f"Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path)
    model.summary()

    # Convert to TFLite with post-training dynamic-range quantization
    # (reduces size by ~2-4x while keeping accuracy close to the original).
    #
    # Bidirectional LSTMs use dynamic TensorList ops that pure TFLite builtins
    # cannot represent, so we enable SELECT_TF_OPS to fall back to full TF
    # kernels for those ops, and disable the experimental lowering pass.
    print("\nConverting to TFLite (with default optimizations + Select TF ops) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    # Write the .tflite file
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_size_mb = os.path.getsize(keras_path) / (1024 * 1024)
    tflite_size_mb = len(tflite_model) / (1024 * 1024)

    print(f"\n✅ Conversion complete!")
    print(f"   Original Keras model : {original_size_mb:.2f} MB  ({keras_path})")
    print(f"   TFLite model         : {tflite_size_mb:.2f} MB  ({tflite_path})")
    print(f"   Size reduction       : {(1 - tflite_size_mb / original_size_mb) * 100:.1f}%")


if __name__ == "__main__":
    convert()
