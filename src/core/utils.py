from typing import Any, Dict, TYPE_CHECKING, List, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import numpy as np
import sys
if TYPE_CHECKING:
    from src.rl_env.rwg_env import RWG_Enviroment
import math
import uuid
from typing import Any, Dict, List
import torch
import torch.nn as nn
import numpy as np
from src.core import config
import re
import zlib
import math
from collections import Counter
from typing import Any, Dict, List
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
class EmbeddingEncoder:
    def __init__(self, model_name: Optional[str] = None, dim: int = 384):
        self.dim = dim
        self.model = None
        if model_name is None:
            model_name = getattr(config, "EMBED_MODEL_NAME", None)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name) if model_name else SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.model = None

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.model:
            try:
                return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            except Exception:
                pass
        embs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            embs.append(rng.standard_normal(self.dim, dtype=np.float32))
        return np.stack(embs, axis=0)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.dim)

_EMBEDDER_INSTANCE = None

def embed(text: str) -> np.ndarray:
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = EmbeddingEncoder()
    
    embedder = _EMBEDDER_INSTANCE
    if not text or not text.strip():
        # Return zero vector for empty text
        dim = embedder.get_sentence_embedding_dimension()
        return np.zeros((dim,), dtype=np.float32)
    try:
        return embedder.encode([text])[0]
    except Exception:
        # Fallback to zero vector on error
        dim = embedder.get_sentence_embedding_dimension()
        return np.zeros((dim,), dtype=np.float32)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def _data_dir() -> str:
    # RWG_DATA_DIR cho phép trỏ sang bộ khác (vd data_multinews) mà không sửa code;
    # mặc định giữ nguyên data2 (OARelatedWork).
    return os.getenv("RWG_DATA_DIR", "data2")


def load_graph_out() -> dict[int, list[int]]:
    path = ROOT / _data_dir() / "citation_adj_out.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        graph = {}
        for k, v in raw.items():
            graph[int(k)] = [int(x) for x in v]
        return graph
    except Exception:
        return {}

def load_graph_in() -> dict[int, list[int]]:
    path = ROOT / _data_dir() / "citation_adj_in.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        graph = {}
        for k, v in raw.items():
            graph[int(k)] = [int(x) for x in v]
        return graph
    except Exception:
        return {}

def load_papers(limit_ids: Optional[Set[int]] = None) -> dict[int, dict]:
    path = ROOT / _data_dir() / os.getenv("RWG_PAPERS_FILE", "OARelatedWork_full.jsonl")
    idx: dict[int, dict] = {}
    if not path.exists():
        return idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            pid = int(p["id"])
            if limit_ids is not None and pid not in limit_ids:
                continue
            structure = ["abstract"] if not p.get("abstract") and not p.get("related_work") else []
            if p.get("abstract"): structure.append("abstract")
            if p.get("related_work"): structure.append("related_work")
            idx[pid] = {
                "id": pid, "title": p.get("title", ""),
                "abstract": p.get("abstract", "") or "",
                "related_work": p.get("related_work", "") or "",
                "structure": structure or ["abstract"]
            }
    return idx

def cos( v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))

def clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

def to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def extract_json(text: str):
    import re, json
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())

def to_json(response: str) -> Dict[str, Any]:
    if not response or not response.strip():
        # Return fallback json if response is empty
        return {"score": "0/5", "reason": "LLM returned empty response or rate limit exceeded"}
        
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1)
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            result = {"score": str(result), "reason": "LLM returned non-dictionary JSON"}
    except Exception:
        result = {"score": "0/5", "reason": "Failed to parse json. Raw: " + response[:50]}
    return result

def safe_text(x: Any) -> str:
    return to_text(x).strip() if x is not None else ""


def tokenize(text: str) -> List[str]:
    text = safe_text(text).lower()
    return re.findall(r"\w+", text)


def length_words(text: str) -> int:
    return max(1, len(tokenize(text)))


def compress_len(text: str) -> float:
    text = safe_text(text)
    if not text:
        return 0.0
    return float(len(zlib.compress(text.encode("utf-8"))))


def conditional_complexity(x: str, y: str) -> float:
    """
    Approximate C(x|y) ≈ C(y + x) - C(y)
    """
    x = safe_text(x)
    y = safe_text(y)
    cy = compress_len(y)
    cyx = compress_len(y + "\n" + x)
    return max(0.0, cyx - cy)


def ncd(x: str, y: str) -> float:
    """
    Normalized Compression Distance
    """
    x = safe_text(x)
    y = safe_text(y)
    if not x and not y:
        return 0.0
    cx = compress_len(x)
    cy = compress_len(y)
    cxy = compress_len(x + "\n" + y)
    denom = max(cx, cy, 1.0)
    return float(max(0.0, min(2.0, (cxy - min(cx, cy)) / denom)))


def semantic_sim(a: str, b: str, skip_embed: bool = False) -> float:
    if skip_embed:
        return 0.0
    ea = embed(safe_text(a))
    eb = embed(safe_text(b))
    return float(cos(ea, eb))


def semantic_dist01(a: str, b: str, skip_embed: bool = False) -> float:
    s = semantic_sim(a, b, skip_embed=skip_embed)
    s01 = (s + 1.0) / 2.0
    return float(1.0 - s01)


def token_distribution(text: str) -> Dict[str, float]:
    toks = tokenize(text)
    if not toks:
        return {"<empty>": 1.0}
    cnt = Counter(toks)
    total = float(sum(cnt.values()))
    return {k: v / total for k, v in cnt.items()}


def kl_div(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-8) -> float:
    keys = set(p.keys()) | set(q.keys())
    val = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        if pk > 0:
            val += pk * math.log((pk + eps) / (qk + eps))
    return float(max(0.0, val))

def js_div(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-8) -> float:
    """
    Jensen-Shannon Divergence.
    Safe against empty distributions and non-overlapping keys.
    Returns value in range [0, log(2)].
    """
    keys = set(p.keys()) | set(q.keys())
    m = {}
    for k in keys:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))
    
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)


def compression_density_reward(text: str) -> float:
    """
    Scientific text has patterns => high compression => low C/Length ratio.
    Target ratio ~0.4-0.7 for good English text.
    Noise ratio ~1.0+.
    We reward LOW ratio (high structure).
    """
    L = len(safe_text(text).encode("utf-8"))
    if L < 10: return 0.0
    c = compress_len(text)
    ratio = c / L
    # Reward peaks when ratio is low (highly structured)
    return float(max(0.0, 1.0 - ratio))


def sigmoid_scalar(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))

def get_unique_cited_ids(text: str) -> Set[str]:
    """Extract unique citation IDs from text.

    Handles single ids ([ID], {ID}, (ID)) AND multi-id groups ([ID1, ID2],
    [ID1; ID2]). Bracket groups containing only digits/space/comma/semicolon are
    scanned for every numeric id — otherwise multi-id citations like [12, 34]
    were silently dropped, inflating the apparent citation-validity rate."""
    if not text:
        return set()
    ids: Set[str] = set()
    for grp in re.findall(r"[\[{(]([\d\s,;]+)[\]})]", text):
        ids.update(re.findall(r"\d+", grp))
    return ids