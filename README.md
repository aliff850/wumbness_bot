# Wumbness Cyberbullying Detection API & Chatbots

<div align="center" >
  <img width="120" height="148" alt="WumboHappi" src="https://github.com/user-attachments/assets/560ed94e-7cbe-4eaa-a1a4-ecfe7a3c7a39" />
  <img width="450" height="146" alt="WumbnessFullLogo" src="https://github.com/user-attachments/assets/44013cdd-1c55-42e8-b5bb-a10fd6285d48" />
</div>
<br>

<p style="text-align: justify;">
  This project is part of our CAI20303 Machine Learning final project where we developed a machine learning solution for cyberbullying detection known as "Wumbness". This project provides a fully decoupled deployment suite for a bidirectional LSTM model trained on the Jigsaw Toxic Comment classification dataset (https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). The setup wraps the model in a FastAPI inference service and connects both a Discord bot as lightweight client services.
</p>

---

## 📂 Directory Structure

All services, scripts, models, and config files are isolated within the `finalproj/` directory:

```text
project_directory/
├── api/
│   ├── models/
│   │   ├── optimized_nostop_bidirectional_lstm_model_10epoch.onnx  # Exported ONNX model
│   │   └── word_index.json                                         # Tokenizer dictionary
│   ├── __init__.py
│   ├── main.py                                                     # FastAPI service entry & routing
│   ├── schemas.py                                                  # Pydantic schemas (Request/Response validation)
│   └── predictor.py                                                # Core ML predictor class using ONNX Runtime
│
├── discord_bot/
│   ├── __init__.py
│   ├── bot.py                                                      # Discord bot client & Supabase integration
│   ├── keep_alive.py                                               # Dummy web server for Render deployment
│   └── requirements.txt                                            # Bot-specific dependencies
│
├── .env                                                            # Environment configuration & API keys
├── requirements.txt                                                # Project dependencies list
└── README.md                                                       # Setup and deployment documentation
```

---

## 🛠️ Architecture & Design Decisions

1. **Decoupled Architecture**: The machine learning model is loaded exactly once into memory by the FastAPI application. The Discord bot runs as an independent asynchronous client script, interacting with the FastAPI server via lightweight HTTP POST requests.
2. **ONNX Runtime Optimization**: To bypass the heavy footprint and deserialization issues of TensorFlow/Keras, the trained model was exported to the .onnx format. Inference is handled by onnxruntime, resulting in a faster, more lightweight service perfectly suited for free-tier cloud environments.
3. **Cloud-Native Deployment**: The API and the Discord bot are designed to be deployed as separate Web Services on Render. A custom keep_alive threaded server ensures the Discord bot satisfies Render's port-binding requirements without needing a paid Background Worker tier.
4. **Persistent Moderation Logging**: Instead of relying on volatile local storage, the bot integrates directly with a Supabase PostgreSQL database to log cyberbullying offenses and generate real-time metrics for server moderators.

---

## 🚀 Setup & Execution

### 1. Virtual Environment & Installation

Create a virtual environment using Python 3.10 to 3.13 (TensorFlow does not currently support Python 3.14+).

#### Option A: Using `uv` (Recommended)
1. Create a virtual environment using Python 3.13:
   ```bash
   uv venv --python 3.13
   ```
2. Activate the virtual environment:
   * **PowerShell**: `.venv\Scripts\activate`
   * **Cmd**: `.venv\Scripts\activate.bat`
   * **Bash/Zsh**: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

#### Option B: Using standard Python & `pip`
1. Create a virtual environment (ensure you are using Python 3.10 - 3.13):
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   * **PowerShell**: `.venv\Scripts\Activate.ps1`
   * **Cmd**: `.venv\Scripts\activate.bat`
   * **Bash/Zsh**: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configure Credentials (`.env`)
Create/edit the `.env` file under the project directory with your bot tokens and configurations:

```env
# Discord Token (obtain from Discord Developer Portal)
DISCORD_TOKEN=your_discord_token_here

# FastAPI Configuration
API_URL=[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
MODEL_PATH=api/models/optimized_nostop_bidirectional_lstm_model_10epoch.onnx
TOKENIZER_PATH=api/models/word_index.json

# Supabase Settings (for moderation logging)
SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Start the Inference Server
Run the FastAPI application locally using Uvicorn:

* **Using `uv`:**
  ```bash
  uv run uvicorn api.main:app --reload --port 8000
  ```
* **Using standard activated environment:**
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```

- Access the API homepage at: `http://127.0.0.1:8000/`
- Test predictions using the interactive Swagger UI: `http://127.0.0.1:8000/docs`

### 4. Launch the Bot Clients
In separate terminal sessions, start the bots:

- **Discord Bot**:
  * **Using `uv`:** `uv run python bots/discord_bot.py`
  * **Using activated env:** `python bots/discord_bot.py`
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

#### Moderation Commands
The bot includes universally accessible commands to track cyberbullying metrics, utilizing modern Discord embeds for a clean UI.

1. **!stats**: Displays high-level server metrics, including the total number of flagged messages and a leaderboard of the top offenders.
2. **!history @user**: Pulls the specific warning log for a mentioned user, utilizing dynamic color coding (green for clean, red for severe offenders) and displaying their most recent offenses. If no user is mentioned, it defaults to the requester's history.
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
