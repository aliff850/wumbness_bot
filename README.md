# Wumbness Cyberbullying Detection API & Chatbots

<div align="center" >
  <img width="120" height="148" alt="WumboHappi" src="https://github.com/user-attachments/assets/560ed94e-7cbe-4eaa-a1a4-ecfe7a3c7a39" />
  <img width="450" height="146" alt="WumbnessFullLogo" src="https://github.com/user-attachments/assets/44013cdd-1c55-42e8-b5bb-a10fd6285d48" />
</div>
<br>

<p style="text-align: justify;">
  This project is part of our CAI20303 Machine Learning final project where we developed a machine learning solution for cyberbullying detection known as "Wumbness". This project provides a fully decoupled deployment suite for a bidirectional LSTM model trained on the Jigsaw Toxic Comment classification dataset (https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). The setup wraps the model in a FastAPI inference service and connects both a Discord bot and a Telegram bot as lightweight client services.
</p>

---

## 📂 Directory Structure

All services, scripts, models, and config files are isolated within the `finalproj/` directory:

```text
finalproj/
├── optimized_nostop_bidirectional_lstm_model.keras   # Trained bidirectional LSTM model
├── keras_tokenizer.pkl                              # Pickled Keras tokenizer
├── lstm.py                                          # Original inference script
│
├── api/
│   ├── __init__.py
│   ├── main.py                                      # FastAPI service entry & routing
│   ├── schemas.py                                   # Pydantic schemas (Request/Response validation)
│   └── predictor.py                                 # Core ML predictor class & Keras 3 monkeypatch
│
├── bots/
│   ├── __init__.py
│   ├── discord_bot.py                               # Discord bot client script
│   └── telegram_bot.py                              # Telegram bot client script
│
├── .env                                             # Environment configuration & bot tokens
├── requirements.txt                                 # Project dependencies list
└── README.md                                        # Setup and deployment documentation
```

---

## 🛠️ Architecture & Design Decisions

1. **Decoupled Architecture**: The CPU-heavy Keras model is loaded exactly once into memory by the FastAPI application. The Discord and Telegram bots run as independent asynchronous client scripts, interacting with the FastAPI server via lightweight HTTP POST requests.
2. **Keras 3 Deserialization Compatibility**: Models trained in older Keras/TensorFlow versions may contain fields like `quantization_config` that cause deserialization errors in newer Keras environments. We dynamically intercept and recursively clean these fields during model initialization to ensure compatibility.
3. **Non-Blocking Prediction Threads**: Because model predictions are CPU-bound, FastAPI endpoint handlers are declared synchronously using `def` rather than `async def`. FastAPI automatically schedules synchronous routes in an external thread pool, preventing chat bot client connections from blocking the server.

---

## 🚀 Setup & Execution

### 1. Installation
Ensure you are in the `finalproj/` directory, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials (`.env`)
Create/edit the `.env` file under `finalproj/` with your bot tokens and configurations:

```env
# Discord Token (obtain from Discord Developer Portal)
DISCORD_TOKEN=your_discord_token_here

# Telegram Token (obtain from BotFather)
TELEGRAM_TOKEN=your_telegram_token_here

# FastAPI Configuration
API_URL=http://127.0.0.1:8000/predict
MODEL_PATH=optimized_nostop_bidirectional_lstm_model.keras
TOKENIZER_PATH=keras_tokenizer.pkl
```

### 3. Start the Inference Server
Run the FastAPI application locally using Uvicorn:
```bash
uvicorn api.main:app --reload --port 8000
```
- Access the API homepage at: `http://127.0.0.1:8000/`
- Test predictions using the interactive Swagger UI: `http://127.0.0.1:8000/docs`

### 4. Launch the Bot Clients
In separate terminal sessions, start the bots:

- **Discord Bot**:
  ```bash
  python bots/discord_bot.py
  ```
- **Telegram Bot**:
  ```bash
  python bots/telegram_bot.py
  ```

---

## 🤖 Bot Operations & Workflows

Both chat bots are designed to intercept user messages asynchronously and query the central FastAPI backend to check for toxicity levels without blocking the chat.

### Discord Bot (`bots/discord_bot.py`)
*   **Library**: `discord.py`
*   **Gateway Intent**: Requires the **Message Content Intent** enabled in your Discord Developer Portal.
*   **Flow**:
    1. Listens to text message events on the server through `on_message`.
    2. Ignores its own messages to prevent warning loops.
    3. Posts the message content asynchronously to `http://localhost:8000/predict` using `httpx`.
    4. If the response flags the message (`is_cyberbullying: true`), it issues a warning in the same channel, mentioning the author and listing the offending categories.

### Telegram Bot (`bots/telegram_bot.py`)
*   **Library**: `python-telegram-bot`
*   **Flow**:
    1. Uses long-polling via `Application.run_polling()` to fetch incoming Telegram updates.
    2. Uses a `MessageHandler` configured with text filters to capture group chat text while ignoring command messages (like `/start`).
    3. Forwards text payloads to the FastAPI prediction endpoint using `httpx`.
    4. If flagged as cyberbullying, it replies directly to the offending message in the group chat notifying the user.

---

## 🔒 API Endpoints

### `POST /predict`
Submits text for cyberbullying classification.

*   **Request Payload**:
    ```json
    {
      "text": "Your message text here"
    }
    ```

*   **Response Payload**:
    ```json
    {
      "text": "Your message text here",
      "predictions": {
        "toxic": 0.00249,
        "severe_toxic": 0.00000,
        "obscene": 0.00001,
        "threat": 0.00000,
        "insult": 0.00002,
        "identity_hate": 0.00000
      },
      "is_cyberbullying": false,
      "detected_categories": []
    }
    ```
