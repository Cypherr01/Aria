"""Reset SQLite and ChromaDB for development."""
import os, shutil, asyncio
from db.database import init_db

DB_PATH = os.getenv("ARIA_DB_PATH", "./data/aria.db")
CHROMA_PATH = os.getenv("ARIA_CHROMA_PATH", "./data/chroma")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Removed {DB_PATH}")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print(f"Removed {CHROMA_PATH}")
os.makedirs("data/chroma", exist_ok=True)
asyncio.run(init_db())
print("✓ Database reset complete.")
