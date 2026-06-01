import os
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environmental variables from the finalproj directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Only inspect text messages and avoid bot loops
    if not update.message or not update.message.text:
        return

    text = update.message.text
    user = update.message.from_user.username or update.message.from_user.first_name

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, json={"text": text}, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data["is_cyberbullying"]:
                    categories = ", ".join(data["detected_categories"])
                    await update.message.reply_text(
                        f"⚠️ **Warning @{user}:** Your message was flagged for cyberbullying "
                        f"({categories}). Please keep the chat respectful and friendly!"
                    )
            else:
                print(f"FastAPI Server Error: Status Code {response.status_code}")
        except Exception as e:
            print(f"Failed to query FastAPI prediction server: {e}")

def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_token_here":
        print("Error: TELEGRAM_TOKEN not set in finalproj/.env")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Intercept all text messages (excluding commands)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_message))
    
    print("Telegram Bot starting polling...")
    print(f"Connecting to FastAPI at: {API_URL}")
    application.run_polling()

if __name__ == "__main__":
    main()
