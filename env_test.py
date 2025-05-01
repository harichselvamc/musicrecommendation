from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

print(f"Client ID: {CLIENT_ID}")
print(f"Client Secret: {CLIENT_SECRET}")
print(f"Redirect URI: {REDIRECT_URI}")
