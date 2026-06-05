"""Run one decay cycle manually. In production: run via cron daily."""
import asyncio, os
from memory.memory_decay import MemoryDecay
from db.repositories.memory_repo import MemoryRepository

async def main():
    repo = MemoryRepository(os.getenv("ARIA_DB_PATH", "./data/aria.db"))
    decay = MemoryDecay(repo, os.getenv("ARIA_CHROMA_PATH", "./data/chroma"))
    result = await decay.run_decay_cycle()
    print(f"Decay cycle complete: {result}")

asyncio.run(main())
