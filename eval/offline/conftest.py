"""
Offline eval config.
Sets a fake OPENAI_API_KEY so modules that instantiate ChatOpenAI at import time
don't fail with a credentials error. No real API calls are made in offline eval.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "offline-test-fake-key")
