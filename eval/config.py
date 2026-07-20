"""
Centralised eval configuration.
All thresholds and file paths live here — no more hardcoding in runner.py or main.py.
Override any value via EVAL_* environment variables or .env file.
"""
from pydantic_settings import BaseSettings


class EvalSettings(BaseSettings):
    # App connection
    APP_URL: str = "http://localhost:8000"
    JWT_SECRET: str = "change-me-in-prod"
    EVAL_TIMEOUT_MS: int = 30000

    # Scoring thresholds — same values used in main.py escalation logic and runner.py pass/fail
    FAITHFULNESS_THRESHOLD: float = 0.7
    COMPLETENESS_THRESHOLD: float = 0.6
    CORRECTNESS_THRESHOLD: float = 0.6
    RAG_PRECISION_THRESHOLD: float = 0.5

    # OpenAI pricing per 1M tokens — update here when OpenAI changes rates
    GPT4O_MINI_INPUT_PRICE:  float = 0.15
    GPT4O_MINI_OUTPUT_PRICE: float = 0.60
    GPT4O_INPUT_PRICE:       float = 2.50
    GPT4O_OUTPUT_PRICE:      float = 10.00
    AVG_PROMPT_TOKENS:       int   = 10_000  # typical prompt size for this bot

    # Regression thresholds — how much degradation vs baseline is acceptable before blocking a PR
    LATENCY_REGRESSION_THRESHOLD: float = 0.2   # 20% latency increase allowed
    COST_REGRESSION_THRESHOLD: float = 0.3       # 30% cost increase allowed
    PASS_RATE_REGRESSION_THRESHOLD: float = 0.0  # any drop in pass rate blocks PR

    # File paths
    BASELINE_FILE: str = "eval/baselines/latest.json"
    DATASET_FILE: str = "eval/dataset.json"

    class Config:
        env_prefix = "EVAL_"
        env_file = ".env"
        extra = "ignore"


eval_settings = EvalSettings()
