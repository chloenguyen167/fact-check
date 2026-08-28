import os
from pathlib import Path

from google import genai

# Load local .env if present (same convention as config.py)
env_path = Path(__file__).resolve().parent / ".env"
if env_path.is_file():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

try:
    models = client.models.list()

    for m in models:
        print(m.name)

except Exception as e:
    print(e)