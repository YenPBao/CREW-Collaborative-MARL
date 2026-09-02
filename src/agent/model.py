from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, Union
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import openai
from src.core.config import OPENAI_CONFIG,MODEL_ALIASES
import anthropic
from google import genai



DEFAULT_LOCAL_URL = "http://localhost:8000/v1"


def normalize_model_name(model: Optional[Union[str, int]]) -> str:
    if model is None:
        return OPENAI_CONFIG.get("model_name", "gpt-4.1-mini")
    s = str(model).strip()
    return MODEL_ALIASES.get(s, s)


def detect_provider(model_name: str) -> Tuple[str, str]:
    m = model_name.strip()
    low = m.lower()

    if low.startswith("openrouter:"):
        return "openrouter", m.split(":", 1)[1].strip()
    if low.startswith(("local:", "vllm:")):
        return "vllm", m.split(":", 1)[1].strip()
    if low.startswith("gemini:"):
        return "gemini", m.split(":", 1)[1].strip()
    if low.startswith(("claude:", "anthropic:")):
        return "anthropic", m.split(":", 1)[1].strip()

    if low.startswith("gemini-"):
        return "gemini", m
    if low.startswith("claude-"):
        return "anthropic", m

    if low.startswith("openai:"):
        return "openai", m.split(":", 1)[1].strip()

    return "openai", m


class WorkerLLM:
    def __init__(self, model_name: Optional[Union[str, int]] = None, verbose: bool = False):
        self.verbose = verbose
        self.model_name = normalize_model_name(model_name)
        self.provider, self.pure_model = detect_provider(self.model_name)
        self.client = None
        # Token counters (accumulated across all generate() calls)
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._num_calls: int = 0
        self.setup_client()

    def setup_client(self) -> None:
        if self.provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", DEFAULT_LOCAL_URL)
            self.client = openai.OpenAI(base_url=base_url, api_key="not-needed", timeout=3600.0)

        if self.provider == "openai":
            api_key = os.getenv(OPENAI_CONFIG.get("api_key_env", "OPENAI_API_KEY"), "").strip()
            if not api_key:
                print("Warning: OPENAI_API_KEY is not set.")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=600.0)

        if self.provider == "openrouter":
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", "not-needed"),
                timeout=120.0
            )

        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = None


        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = None

    def reset_token_counts(self) -> None:
        """Reset token counters to zero."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._num_calls = 0

    def get_token_counts(self) -> dict:
        """Return accumulated token counts."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "num_calls": self._num_calls,
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> Tuple[str, int]:
        max_new_tokens = max_tokens if max_tokens is not None else int(OPENAI_CONFIG.get("max_output_tokens", 256))
        temperature = float(OPENAI_CONFIG.get("temperature", 0.7))
        top_p = float(OPENAI_CONFIG.get("top_p", 0.95))

        if self.provider in ("openai", "vllm", "openrouter"):
            MAX_CHARS = 100000
            if len(prompt) > MAX_CHARS:
                prompt = prompt[:MAX_CHARS] + "\n...[TRUNCATED to fit context window]"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            print(f"DEBUG WorkerLLM: Sending {len(prompt)} chars to {self.provider} ({self.pure_model})...", flush=True)

            retries = 5
            delay = 2
            for attempt in range(retries):
                try:
                    extra = (
                        {"extra_body": {"provider": {"ignore": ["Novita"]}}}
                        if self.provider == "openrouter"
                        else {}
                    )
                    resp = self.client.chat.completions.create(
                        model=self.pure_model,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **extra,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    usage = getattr(resp, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens
                    new_tokens = completion_tokens if completion_tokens else max(0, total_tokens - prompt_tokens)
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += new_tokens
                    self._total_tokens += total_tokens
                    self._num_calls += 1
                    return text, new_tokens
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = any(x in err_str for x in ["connection", "timeout", "rate limit", "overloaded", "502", "503", "504", "429"])
                    if retryable and attempt < retries - 1:
                        print(f"   WorkerLLM Retry {attempt+1}/{retries} ({self.provider}): {e}. Waiting {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    print(f"[ERROR] WorkerLLM API Error ({self.provider}): {e}")
                    return "", 0

        if self.provider == "anthropic":
            if not self.client:
                print("WorkerLLM API Error (anthropic): Missing ANTHROPIC_API_KEY")
                return "", 0
            system = system_prompt or ""
            retries = 5
            delay = 2
            for attempt in range(retries):
                try:
                    msg = self.client.messages.create(
                        model=self.pure_model,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(
                        block.text for block in msg.content
                        if getattr(block, "type", None) == "text"
                    ).strip()
                    usage = getattr(msg, "usage", None)
                    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                    new_tokens = getattr(usage, "output_tokens", 0) if usage else 0
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += int(new_tokens)
                    self._total_tokens += prompt_tokens + int(new_tokens)
                    self._num_calls += 1
                    return text, int(new_tokens)
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = any(x in err_str for x in ["connection", "timeout", "rate limit", "overloaded", "502", "503", "504", "429"])
                    if retryable and attempt < retries - 1:
                        print(f"   WorkerLLM Retry {attempt+1}/{retries} ({self.provider}): {e}. Waiting {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    print(f"[ERROR] WorkerLLM API Error ({self.provider}): {e}")
                    return "", 0

        if self.provider == "gemini":
            if not self.client:
                print("WorkerLLM API Error (gemini): Missing GEMINI_API_KEY or GOOGLE_API_KEY")
                return "", 0
            merged = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
            retries = 5
            delay = 2
            print(f"DEBUG WorkerLLM: Preparing Gemini call for model {self.pure_model}...")
            for attempt in range(retries):
                try:
                    print(f"DEBUG WorkerLLM: Sending to Gemini (attempt {attempt+1})...")
                    resp = self.client.models.generate_content(model=self.pure_model, contents=merged)
                    text = (getattr(resp, "text", "") or "").strip()
                    usage_meta = getattr(resp, "usage_metadata", None)
                    prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
                    completion_tokens = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
                    total_tokens = getattr(usage_meta, "total_token_count", 0) if usage_meta else (prompt_tokens + completion_tokens)
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens
                    self._total_tokens += total_tokens
                    self._num_calls += 1
                    return text, completion_tokens
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = any(x in err_str for x in ["connection", "timeout", "rate limit", "overloaded", "502", "503", "504", "429"])
                    if retryable and attempt < retries - 1:
                        print(f"   WorkerLLM Retry {attempt+1}/{retries} ({self.provider}): {e}. Waiting {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    print(f"[ERROR] WorkerLLM API Error ({self.provider}): {e}")
                    return "", 0


    def get_embedding(self, texts: List[str]) -> np.ndarray:
        if self.provider not in ("openai", "vllm"):
            raise RuntimeError("Embeddings only supported for openai/vllm in this wrapper.")
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([item.embedding for item in resp.data], dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 1536