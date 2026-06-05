#!/bin/bash
echo "Setting up ARIA..."
pip install -r requirements.txt
python -m spacy download en_core_web_sm
mkdir -p data/uploads data/chroma
python -c "import asyncio; from db.database import init_db; asyncio.run(init_db())"
echo "✓ Setup complete. Copy .env.example to .env and add your API keys."
