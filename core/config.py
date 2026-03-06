"""
config.py — Loads all settings from .env
Single source of truth for configuration across the entire agent.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # ── vLLM (primary inference engine) ──────────────────────────────────
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    # ── Ollama (embeddings only) ──────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Models ────────────────────────────────────────────────────────────
    # Must match exactly what vLLM is serving (check `curl localhost:8000/v1/models`)
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    FAST_MODEL:    str = os.getenv("FAST_MODEL",    "Qwen/Qwen2.5-7B-Instruct")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # ── Parallelism ───────────────────────────────────────────────────────
    # vLLM handles batching internally. This caps concurrent HTTP connections.
    # RTX 6000 Pro 95GB + Qwen2.5 7B fp8: 20 workers, vLLM batches them.
    PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "20"))

    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_DIR:          Path = Path(__file__).parent.parent
    DATA_DIR:          Path = BASE_DIR / os.getenv("DATA_DIR",           "data")
    UPLOADS_DIR:       Path = BASE_DIR / os.getenv("UPLOADS_DIR",        "data/uploads")
    PROCESSED_DIR:     Path = BASE_DIR / os.getenv("PROCESSED_DIR",      "data/processed")
    KNOWLEDGE_BASE_DIR: Path = BASE_DIR / os.getenv("KNOWLEDGE_BASE_DIR","data/knowledge_base")
    PLAYBOOK_PATH:     Path = BASE_DIR / os.getenv("PLAYBOOK_PATH",      "data/knowledge_base/playbook.yaml")

    # ── Chunking ──────────────────────────────────────────────────────────
    MAX_CHUNK_TOKENS:     int = int(os.getenv("MAX_CHUNK_TOKENS",     512))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS",  50))

    # ── Agent ─────────────────────────────────────────────────────────────
    RISK_THRESHOLD: str = os.getenv("RISK_THRESHOLD", "MEDIUM")

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str  = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE:  Path = BASE_DIR / os.getenv("LOG_FILE", "data/agent.log")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.UPLOADS_DIR, cls.PROCESSED_DIR, cls.KNOWLEDGE_BASE_DIR]:
            d.mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_dirs()