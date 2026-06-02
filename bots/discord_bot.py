import os
import discord
from discord.ext import commands
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
from collections import Counter
from keep_alive import keep_alive

# Load environmental variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

intents = discord.Intents.default()
intents.message_content = True  # Required to read text message contents

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Discord Bot successfully logged in as: {bot.user}")
    print(f"Connecting to FastAPI at: {API_URL}")
    print("Connected to Supabase successfully.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

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
                    
                    # === NEW: Push data to Supabase ===
                    try:
                        supabase.table("warnings").insert({
                            "user_id": str(message.author.id),
                            "username": str(message.author.name),
                            "categories": categories,
                            "message_content": message.content  # <-- ADD THIS LINE
                        }).execute()
                    except Exception as e:
                        print(f"Failed to log to Supabase: {e}")
                    # ==================================

                    await message.channel.send(
                        f"⚠️ **Warning {message.author.mention}:** Your message was flagged for cyberbullying "
                        f"({categories}). Please keep the conversation respectful!"
                    )
            else:
                print(f"FastAPI Server Error: Status Code {response.status_code}")
        except Exception as e:
            print(f"Failed to query FastAPI prediction server: {e}")

    # Ensure commands still process after reading the message
    await bot.process_commands(message)

# ==========================================
# UNIVERSALLY ACCESSIBLE MODERATION COMMANDS
# ==========================================

@bot.command(name="stats")
async def server_stats(ctx):
    """Shows overall cyberbullying metrics for the server in a beautiful embed."""
    
    # Fetch all records from Supabase
    response = supabase.table("warnings").select("user_id, username").execute()
    records = response.data
    total_warnings = len(records)
    
    # 1. Layout for a completely clean server
    if total_warnings == 0:
        clean_embed = discord.Embed(
            title="✨ Server is Clean!",
            description="Excellent news! No toxic messages have been logged yet.",
            color=discord.Color.green() # Bright green color
        )
        await ctx.send(embed=clean_embed)
        return

    # Count warnings per username using Python's Counter
    user_counts = Counter(row["username"] for row in records)
    top_offenders = user_counts.most_common(5)
    
    # 2. Modern Embed Design for Stats
    embed = discord.Embed(
        title="🛡️ Cyberbullying Metrics", 
        description="Here is the current moderation overview for this server.",
        color=0x5865F2 # The official modern Discord "Blurple" color
    )
    
    # Add the server's icon as a thumbnail in the top right (if the server has one)
    if ctx.guild.icon:
        embed.set_thumbnail(url=ctx.guild.icon.url)
        
    # Use code-block highlighting for numbers to make them pop
    embed.add_field(name="Total Messages Flagged", value=f"` {total_warnings} `", inline=False)
    
    # Format offenders beautifully with emojis based on their rank
    offenders_text = ""
    for i, (name, count) in enumerate(top_offenders, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "👤"
        offenders_text += f"{medal} **{name}** — `{count} warnings`\n"
        
    embed.add_field(name="Top Offenders", value=offenders_text, inline=False)
    
    # Add a professional footer
    embed.set_footer(
        text=f"Requested by {ctx.author.name}", 
        icon_url=ctx.author.display_avatar.url
    )
    
    await ctx.send(embed=embed)

@bot.command(name="history")
async def check_user(ctx, member: discord.Member = None):
    """Checks the warning history of a specific user with dynamic color styling."""
    
    # If no one is tagged, default to the person who sent the command
    if member is None:
        member = ctx.author
    
    response = supabase.table("warnings") \
        .select("categories, timestamp") \
        .eq("user_id", str(member.id)) \
        .order("timestamp", desc=True) \
        .execute()
        
    records = response.data
    
    # 1. Layout for a clean user
    if not records:
        clean_embed = discord.Embed(
            title="✅ Clean Record",
            description=f"{member.mention} has a perfect record with no warnings.",
            color=discord.Color.green()
        )
        await ctx.send(embed=clean_embed)
        return
        
    # 2. Dynamic color: Red for severe offenders (3+ warnings), Yellow for others
    embed_color = 0xED4245 if len(records) >= 3 else 0xFEE75C
    
    embed = discord.Embed(
        title=f"📝 Moderation History", 
        description=f"Warning log for {member.mention}",
        color=embed_color
    )
    
    # Display the user's Discord avatar in the top right corner
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.add_field(name="Total Warnings", value=f"` {len(records)} `", inline=False)
    
    recent_history = ""
    for row in records[:5]:
        date_str = row["timestamp"].split("T")[0] 
        # Format the categories into nice inline code blocks (e.g., `toxic`, `insult`)
        formatted_categories = " ".join([f"`{c.strip()}`" for c in row['categories'].split(",")])
        recent_history += f"**{date_str}:** {formatted_categories}\n"
        
    embed.add_field(name="Recent Offenses (Last 5)", value=recent_history, inline=False)
    
    # Put their raw Discord ID in the footer for moderation logging purposes
    embed.set_footer(text=f"User ID: {member.id}")
    
    await ctx.send(embed=embed)

if __name__ == "__main__":
    if not DISCORD_TOKEN or DISCORD_TOKEN == "your_discord_token_here":
        print("Error: DISCORD_TOKEN not set")
    else:
        keep_alive()
        bot.run(DISCORD_TOKEN)
