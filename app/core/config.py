import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if DATABASE_URL is None:
    raise RuntimeError(f"DATABASE_URL introuvable — vérifie que .env existe bien à {env_path}")
if GEMINI_API_KEY is None:
    raise RuntimeError(f"GEMINI_API_KEY introuvable — vérifie que .env existe bien à {env_path}")