import joblib
import json

# Load your pickled Keras tokenizer
tokenizer = joblib.load("keras_tokenizer_v2.pkl")

# Save just the word dictionary as a lightweight JSON
with open("word_index.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer.word_index, f)