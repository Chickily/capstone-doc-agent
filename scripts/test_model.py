import time
from src.config import settings, get_secret
from src.llm import LLM

llm = LLM(get_secret(settings.api_key_name), settings.model, settings.provider)
t = time.time()
try:
    out = llm.complete("Reply briefly.", [{"role": "user", "content": "Say OK"}])
    print("OK  ", settings.model, round(time.time() - t, 1), "s ->", out[:40])
except Exception as e:
    print("FAIL", settings.model, type(e).__name__, str(e)[:160])
