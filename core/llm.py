"""
llm.py — vLLM + Ollama hybrid client

LLM inference  → vLLM (OpenAI-compatible, continuous batching, true parallelism)
Embeddings     → Ollama (unchanged, nomic-embed-text)

vLLM continuous batching means ALL concurrent clause requests get processed
in overlapping forward passes — real GPU parallelism, not a queue.

On RTX 6000 Pro 95GB with Qwen2.5 7B fp8:
  ~2500 tok/s aggregate → 46 clauses in ~8-12s → ~0.2-0.3s effective per clause
"""

import asyncio
from typing import Optional, Generator

import httpx
from openai import OpenAI, AsyncOpenAI
import ollama
from loguru import logger

from core.config import config


# ── Shared async HTTP client for embeddings ───────────────────────────────
_async_http: Optional[httpx.AsyncClient] = None

def _get_http() -> httpx.AsyncClient:
    global _async_http
    if _async_http is None or _async_http.is_closed:
        _async_http = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=30),
        )
    return _async_http


# ------------------------------------------------------------------
# Async LLM Client — vLLM OpenAI-compatible endpoint
# ------------------------------------------------------------------

class AsyncLLMClient:
    """
    Async client for vLLM.
    vLLM's OpenAI-compatible server handles continuous batching internally —
    just fire all requests concurrently and it batches them on the GPU.
    No semaphore needed: vLLM manages its own queue efficiently.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=config.VLLM_BASE_URL,
            api_key="not-needed",          # vLLM doesn't require auth
            timeout=300.0,
            max_retries=2,
        )
        self.primary_model = config.PRIMARY_MODEL
        self.fast_model    = config.FAST_MODEL
        # Semaphore: prevent hammering vLLM with more than N simultaneous requests
        # vLLM handles batching internally but too many concurrent connections
        # can overwhelm the HTTP layer. 20 is safe on your hardware.
        self._sem = asyncio.Semaphore(config.PARALLEL_WORKERS)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        use_model = model or self.primary_model
        messages  = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"vLLM async generation failed ({use_model}): {e}")
                raise

    async def fast_generate(self, prompt: str, system: Optional[str] = None) -> str:
        return await self.generate(
            prompt=prompt,
            system=system,
            model=self.fast_model,
            temperature=0.0,
            max_tokens=512,
        )


# ------------------------------------------------------------------
# Sync LLM Client — vLLM OpenAI-compatible endpoint
# Used for: metadata extraction, executive summary (called from sync context)
# ------------------------------------------------------------------

class LLMClient:
    """
    Sync client for vLLM.
    Embeddings still go through Ollama (nomic-embed-text not served by vLLM).
    """

    def __init__(self):
        # vLLM for text generation
        self.client = OpenAI(
            base_url=config.VLLM_BASE_URL,
            api_key="not-needed",
            timeout=300.0,
            max_retries=2,
        )
        # Ollama for embeddings only
        self.ollama_client   = ollama.Client(host=config.OLLAMA_BASE_URL)
        self.primary_model   = config.PRIMARY_MODEL
        self.fast_model      = config.FAST_MODEL
        self.embedding_model = config.EMBEDDING_MODEL

    def check_connection(self) -> bool:
        """Check both vLLM and Ollama are reachable."""
        ok = True

        # Check vLLM
        try:
            models    = self.client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"vLLM connected. Models: {available}")
            if self.primary_model not in available and not any(self.primary_model in m for m in available):
                logger.warning(f"Model '{self.primary_model}' not found in vLLM. Available: {available}")
                ok = False
            else:
                logger.success(f"vLLM ready — {self.primary_model}")
        except Exception as e:
            logger.error(f"Cannot connect to vLLM at {config.VLLM_BASE_URL}: {e}")
            logger.error("Start vLLM with: bash start_vllm.sh")
            ok = False

        # Check Ollama (for embeddings)
        try:
            self.ollama_client.list()
            logger.success("Ollama ready — embeddings available")
        except Exception as e:
            logger.warning(f"Ollama not reachable at {config.OLLAMA_BASE_URL}: {e}")
            logger.warning("Embeddings/RAG will be unavailable")
            # Not fatal — embeddings are optional

        return ok

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        use_model = model or self.primary_model
        messages  = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM generation failed ({use_model}): {e}")
            raise

    def fast_generate(self, prompt: str, system: Optional[str] = None) -> str:
        return self.generate(
            prompt=prompt, system=system,
            model=self.fast_model,
            temperature=0.0, max_tokens=512,
        )

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        use_model = model or self.primary_model
        messages  = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for chunk in self.client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # Embeddings still via Ollama
    def embed(self, text: str) -> list[float]:
        try:
            resp = self.ollama_client.embeddings(
                model=self.embedding_model, prompt=text,
            )
            return resp.embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# Singletons
llm       = LLMClient()
async_llm = AsyncLLMClient()