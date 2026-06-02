import os
import discord
from discord.ext import commands
import httpx
from dotenv import load_dotenv
from keep_alive import keep_alive

# Load environmental variables from the finalproj directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

# Discord bot token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "https://wumbness-api.onrender.com/predict")

intents = discord.Intents.default()
intents.message_content = True  # Required to read text message contents

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Discord Bot successfully logged in as: {bot.user}")
    print(f"Connecting to FastAPI at: {API_URL}")

@bot.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == bot.user:
        return

    # Skip empty or embed-only messages
    if not message.content.strip():
        await bot.process_commands(message)
        return

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, json={"text": message.content}, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                if data["is_cyberbullying"]:
                    categories = ", ".join(data["detected_categories"])
                    await message.channel.send(
                        f"⚠️ **Warning {message.author.mention}:** Your message was flagged for cyberbullying "
                        f"({categories}). Please keep the conversation respectful!"
                    )
            else:
                print(f"FastAPI Server Error: Status Code {response.status_code}")
        except Exception as e:
            print(f"Failed to query FastAPI prediction server: {e}")

    await bot.process_commands(message)

if __name__ == "__main__":
    if not DISCORD_TOKEN or DISCORD_TOKEN == "your_discord_token_here":
        print("Error: DISCORD_TOKEN not set")
    else:
        keep_alive()
        bot.run(DISCORD_TOKEN)
