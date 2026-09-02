from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
import gymnasium.spaces as spaces
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.agent.model import WorkerLLM
from src.agent.action import AgentAction, ACTION_DISPATCHER,NUM_ACTIONS
from src.core import config
from src.core.utils import (
    embed, cos, clip01, to_text,
    safe_text, tokenize, length_words, compress_len,
    conditional_complexity, ncd,
    semantic_sim, semantic_dist01,
    token_distribution, kl_div,
    compression_density_reward, sigmoid_scalar,
    load_papers, load_graph_out, load_graph_in
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
from src.core.utils import embed, cos, clip01, to_text, js_div, token_distribution, compression_density_reward

class RewardNormalizer:
    """
    Normalizes rewards using Exponential Moving Average (EMA) of mean and variance.
    Fixes Problem 5 (Z-score NaN by initializing variance to 1.0).
    """
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0  # Initialize with 1.0 to avoid early NaN
        
    def update(self, x: float) -> float:
        # Update EMA mean and variance
        delta = x - self.mean
        self.mean += (1 - self.alpha) * delta
        self.var = self.alpha * self.var + (1 - self.alpha) * (delta ** 2)
        
        # Return normalized score
        std = np.sqrt(max(self.var, self.eps))
        return (x - self.mean) / std
class RWG_Environment:
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_agents: int = 2, llm=None, use_gpu: bool = False, persist_directory: Optional[str] = None, skip_embed: bool = False, skip_reward_embed: bool = False, papers=None, citation_adj_in=None, citation_adj_out=None):
        #setting up
        self.skip_embed = skip_embed
        self.skip_reward_embed = skip_reward_embed
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents_list = self.possible_agents[:]
        # Global workspace
        self.papers: Dict[int, Dict[str, Any]] = papers if papers is not None else load_papers()
        self.global_knowledge: str = ""
        self.draft: str = ""
        # RL state
        self.round: int = 0
        self.max_rounds: int = 20
        # Task info
        self.paper_title: str = ""
        self.abstract: str = ""
        #Token
        self.token_total: int = 0
        self.agent_token_usage: Dict[str, int] = {aid: 0 for aid in self.possible_agents}
        # Track tokens per action type: SEARCH(L), FEEDBACK(F), UPDATE(G), WRITE(D)
        self.action_token_usage: Dict[str, Dict[str, int]] = {
            aid: {"L": 0, "F": 0, "G": 0, "D": 0} for aid in self.possible_agents
        }
        self.read_history: Dict[str, List[Dict[int, str]]] = {aid: [] for aid in self.possible_agents}  # id,section
        # Quality metrics history
        self.quality_history: List[Dict[str, float]] = []
        self.quality_metrics: List[Dict[str, float]] = []
        # Runtime state
        self.current_actions: Dict[str, int] = {}
        self.list_actions: List[Dict[str,int]] = []
        # Per-agent local metadata cache (L)
        self.local_cache: Dict[str, Dict[str, Any]] = {aid: {"memory": "", "rationale": ""} for aid in self.possible_agents}   
        self.feedback: str = ""  # Shared feedback (like global_knowledge)
        # Models & memory
        self.llm = llm if llm is not None else WorkerLLM()
        # Citation graph for graph metrics (initialize as empty, populate from data if available)
        self.citation_adj_in: Dict[int, List[int]] = citation_adj_in if citation_adj_in is not None else load_graph_in()
        self.citation_adj_out: Dict[int, List[int]] = citation_adj_out if citation_adj_out is not None else load_graph_out()
        # Search candidates tracking
        self.latest_candidates: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in self.possible_agents}
        self.search_top_k: int = getattr(config, "SEARCH_TOP_K", 5)
        # Reward weights 
        self.wP, self.wG,  self.wL, self.wF = 1.0, 0.4,  0.3, 0.3
        self.wGraph: float = 0.2  # Weight for graph-based rewards
        self.lam: float = 0.001  # Lambda for token cost penalty
        # Episode stats
        self.episode_stats: Dict[str, Any] = {"actions_taken": {a.name: 0 for a in AgentAction}}
        # Observation and action spaces
        self.obs_space = NUM_ACTIONS
        self.act_space = spaces.Discrete(NUM_ACTIONS)
        # Target 
        self.draft_target = ""
        self.cited_paper : List[Dict[str,str]] = []
        # Parameter
        self.alpha :float = 0.4 
        self.eps = 1e-8
        self.last_writer: str = ""
        self.write_credit_ratio: float = 0.3
        self.draft_ambient_gamma: float = 0.1

        # Reward normalization
        self.reward_norm = {aid: RewardNormalizer() for aid in self.possible_agents}

        # Locks for shared state — used during parallel action execution
        self._draft_lock        = threading.Lock()
        self._global_lock       = threading.Lock()
        self._feedback_lock     = threading.Lock()
        self._cited_lock        = threading.Lock()
        self._token_lock        = threading.Lock()

    @property
    def agent_actions_history(self):
        return self.list_actions
    def update_local_cache(self, aid: str, response: Dict[str, Any]):
        if aid in self.local_cache:
            self.local_cache[aid]["memory"] = response.get("memory", "")
            self.local_cache[aid]["rationale"] = response.get("rationale", "")
        else:
            self.local_cache[aid] = {
                "memory": response.get("memory", ""),
                "rationale": response.get("rationale", "")
            }
    def update_draft(self, aid: str, response: Dict[str, Any]):
        new_content = response.get("related_work", "")
        if not new_content:
            for k, v in response.items():
                if isinstance(v, str) and len(v) > len(new_content):
                    new_content = v
        if new_content:
            if not isinstance(new_content, str):
                new_content = json.dumps(new_content, ensure_ascii=False)
            # Lock: draft là tài nguyên chung, agent nào xong trước thì ghi trước
            with self._draft_lock:
                self.draft = new_content
                self.last_writer = aid
        else:
            print(f"    [Warning] Agent {aid} returned empty draft. Keys: {list(response.keys())}")
    
    def update_tokens(self, agent_id: str, usage: int):
        with self._token_lock:
            if agent_id in self.agent_token_usage:
                self.agent_token_usage[agent_id] += usage
                self.token_total += usage
            else:
                print(f"Warning: Agent {agent_id} not found in token tracking.")
    def observation_space(self, agent: str) -> spaces.Space: return self._obs_space
    def action_space(self, agent: str) -> spaces.Space:       return self._act_space

    def reset(self, seed: int = None, options: Dict[str, Any] = None, clear_memory: bool = False):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)   
        
        if options is None:
            options = {}
        
        # 1. Reset Agents & Rounds
        self.agents_list = self.possible_agents[:]
        self.max_rounds = options.get("max_rounds", self.max_rounds)
        self.round = 0
        
        # 2. Reset Global Workspace
        self.draft = ""
        self.global_knowledge = ""
        
        # 3. Reset Task Info (from options if provided)
        self.paper_title = options.get("paper_title", "")
        self.abstract = options.get("abstract", "")
        
        # 4. Reset Memory & Caches
        self.current_actions = {}
        self.local_cache = {aid: {"memory": "", "rationale": ""} for aid in self.possible_agents}
        self.feedback = ""  # Shared feedback reset
        self.read_history = {aid: [] for aid in self.possible_agents}
        self.latest_candidates = {aid: [] for aid in self.possible_agents}
        self.cited_paper = []
        self.last_writer = ""

        # 5. Reset Accounting & History
        self.token_total = 0
        self.agent_token_usage = {aid: 0 for aid in self.possible_agents}
        self.action_token_usage = {aid: {"L": 0, "F": 0, "G": 0, "D": 0} for aid in self.possible_agents}
        self.quality_history = []
        self.episode_stats = {"actions_taken": {a.name: 0 for a in AgentAction}}
        
        # 6. Re-initialize RL State (Blackboard)
        return self.get_obs(), {aid: {} for aid in self.agents_list}
    
    def get_obs(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}

        # -- 1. Time --------------------------------------------------------------
        t_norm = clip01(self.round / self.max_rounds)

        # -- 2. Action counts -----------------------------------------------------
        total_counts = {a: 0 for a in range(NUM_ACTIONS)}
        agent_counts = {aid: {a: 0 for a in range(NUM_ACTIONS)} for aid in self.possible_agents}
        
        for turn_actions in self.list_actions:
            for aid, act in turn_actions.items():
                if aid in agent_counts:
                    total_counts[act] += 1
                    agent_counts[aid][act] += 1

        # -- 3. Texts -------------------------------------------------------------
        L_text = {a: to_text(self.local_cache.get(a, {}).get("memory", "")) for a in self.possible_agents}
        G_text = self.global_knowledge or ""
        D_text = self.draft or ""
        T_text = (self.paper_title + " " + (self.abstract or "")) if self.paper_title else (self.draft_target or "")

        # -- 4. Embeddings --------------------------------------------------------
        if self.skip_embed:
            dim = 384
            eT = np.zeros(dim, dtype=np.float32)
            eG = np.zeros(dim, dtype=np.float32)
            eD = np.zeros(dim, dtype=np.float32)
            eF_shared = np.zeros(dim, dtype=np.float32)
            eL = {a: np.zeros(dim, dtype=np.float32) for a in self.possible_agents}
        else:
            eT = embed(T_text)
            eG = embed(G_text)
            eD = embed(D_text)
            eF_shared = embed(to_text(self.feedback))
            eL = {a: embed(txt) for a, txt in L_text.items()}

        # -- 5. Action distribution (impact column) -------------------------------
        dist = self.build_action()

        # -- Helpers --------------------------------------------------------------
        def norm_tokens(t: int, budget: float = 2000.0) -> float:
            """Linear token-budget countdown: 1.0 (unused) → 0.0 (exhausted)."""
            return max(0.0, 1.0 - float(t) / budget)

        def safe_clip_social(total: int, mine: int) -> float:
            """Ratio of actions by OTHER agents."""
            if total == 0: return 0.0
            return clip01(float((total - mine) / total))

        # -- 6. Global shared features --------------------------------------------
        relD_global = clip01(float(cos(eD, eT)))

        # Gap vector for feedback alignment
        gap_vec = eT - eD
        gap_norm = float(np.linalg.norm(gap_vec)) + self.eps
        gap_vec_n = gap_vec / gap_norm

        for aid in self.agents_list:
            # -- Row SEARCH (0) ------------------------------------------------
            relL = clip01(float(cos(eL[aid], eT)))
            novL = clip01(1.0 - float(cos(eL[aid], eG)))

            ids_i = self.paper_ids(aid)
            ids_others = set()
            for other in self.possible_agents:
                if other != aid: ids_others |= self.paper_ids(other)
            dup_id = float(len(ids_i & ids_others) / max(1, len(ids_i)))
            sims = [float(cos(eL[aid], eL[other])) for other in self.possible_agents if other != aid]
            dup_sem = float(max(sims)) if sims else 0.0
            covL = clip01(1.0 - (self.alpha * dup_id + (1.0 - self.alpha) * dup_sem))

            impactL = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.SEARCH)]))
            tokL = norm_tokens(self.action_token_usage[aid]["L"])
            socialL = safe_clip_social(total_counts[0], agent_counts[aid][0])

            # -- Row FEEDBACK (3) ----------------------------------------------
            relF_draft = float(cos(eF_shared, eD))
            relF_target = float(cos(eF_shared, eT))
            relF = clip01((1.0 - self.alpha) * relF_draft + self.alpha * relF_target)
            covF = clip01(float(np.dot(eF_shared, gap_vec_n)))

            impactF = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.FEEDBACK)]))
            tokF = norm_tokens(self.action_token_usage[aid]["F"])
            socialF = safe_clip_social(total_counts[3], agent_counts[aid][3])

            # -- Row UPDATE (1) ------------------------------------------------
            relG_agent = clip01(float(cos(eL[aid], eT)))
            novG_agent = clip01(1.0 - float(cos(eL[aid], eG)))

            impactG = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.UPDATE)]))
            tokG = norm_tokens(self.action_token_usage[aid]["G"])
            socialG = safe_clip_social(total_counts[1], agent_counts[aid][1])

            # -- Row WRITE (2) -------------------------------------------------
            relD_row = relD_global
            covD = clip01(float(cos(eD, eG)))

            impactD = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.WRITE)]))
            tokD = norm_tokens(self.action_token_usage[aid]["D"])
            socialD = safe_clip_social(total_counts[2], agent_counts[aid][2])

            # Matrix: [Rel, Nov/Cov, Impact, tok_norm, social]  shape (4, 5)
            mat = np.array([
                [relL,       covL,       impactL, tokL, socialL], # SEARCH
                [relF,       covF,       impactF, tokF, socialF], # FEEDBACK
                [relG_agent, novG_agent, impactG, tokG, socialG], # UPDATE
                [relD_row,   covD,       impactD, tokD, socialD], # WRITE
            ], dtype=np.float32)

            obs[aid] = mat
        return obs
    
    def safe_cos(self, a, b):
        """Cosine similarity safe against zero vectors.
        Returns 0.0 when either vector has near-zero norm (instead of NaN)."""
        a = a.view(1, -1)
        b = b.view(1, -1)
        if a.norm() < self.eps or b.norm() < self.eps:
            return torch.tensor(0.0, device=self.device)
        return F.cosine_similarity(a, b).squeeze()

    def get_q_torch(self, text_emb, target_emb, doc_type="draft", extra_emb=None, aid=None, text_raw=""):
        """
        Calculates Quality Q with fixes for Problems 2, 3, 4.
        """
        if text_emb is None: return torch.tensor(0.0, device=self.device)
        
        text_emb = text_emb.view(1, -1)
        target_emb = target_emb.view(1, -1)

        # Base Relevance (Cosine)
        rel = self.safe_cos(text_emb, target_emb)

        # 1. Draft Quality: Relevance + Compression Consistency
        if doc_type == "draft":
            struct_reward = compression_density_reward(text_raw)
            return 0.8 * rel + 0.2 * struct_reward

        # 2. Local Quality: Rel + Density - Redundancy
        elif doc_type == "local":
            topic_str = getattr(self, "paper_title", "") + " " + getattr(self, "abstract", "")
            topic_emb = torch.from_numpy(embed(topic_str[:1000])).to(self.device).view(1, -1)
            density = self.safe_cos(text_emb, topic_emb) 
            
            penalty = torch.tensor(0.0, device=self.device)
            if aid in self.read_history and self.read_history[aid]:
                hist_list = [to_text(h) for h in self.read_history[aid][-5:]] # Last 5
                if hist_list:
                    history_embs = torch.stack([torch.from_numpy(embed(h)).to(self.device) for h in hist_list])
                    sims = [self.safe_cos(text_emb, history_embs[i:i+1]) for i in range(history_embs.size(0))]
                    if sims: penalty = torch.max(torch.stack(sims))
            
            return 0.5 * rel + 0.3 * density - 0.2 * penalty

        # 3. Global Quality: Relevance + Novelty (JS-Divergence based)
        elif doc_type == "global":
            # Semantic Novelty
            if extra_emb is not None:
                extra_emb = extra_emb.view(1, -1)
                novelty = 1.0 - self.safe_cos(text_emb, extra_emb)
            else:
                novelty = torch.tensor(0.0, device=self.device) # FIX Problem 2: Start at 0 novelty
            
            # Linguistic Diversity (JS Divergence) - FIX Problem 4
            p_dist = token_distribution(text_raw)
            gold_dist = token_distribution(self.draft_target)
            diversity = 1.0 - clip01(js_div(p_dist, gold_dist) / np.log(2))
            
            return 0.5 * rel + 0.3 * novelty + 0.2 * diversity

        # 4. Feedback Quality: Gap-driven
        elif doc_type == "feedback":
            if extra_emb is not None:
                extra_emb = extra_emb.view(1, -1)
                gap_vector = target_emb - extra_emb
                gap_norm = gap_vector.norm()
                if gap_norm < self.eps: return torch.tensor(0.0, device=self.device)
                gap_vector = gap_vector / gap_norm
                return self.safe_cos(text_emb, gap_vector)
            return rel
        return torch.tensor(0.0, device=self.device)

    def step(self, actions: Dict[str, Any]):
        _skip_r = self.skip_embed or self.skip_reward_embed

        # 1. Prepare Target/Draft Embeddings (for reward only — skipped at inference)
        if not _skip_r:
            e_gold = torch.from_numpy(embed(self.draft_target)).to(self.device)
            e_draft_old = torch.from_numpy(embed(self.draft)).to(self.device)
            q_old_draft = self.get_q_torch(e_draft_old, e_gold, "draft")
        else:
            e_gold = torch.zeros(384, device=self.device)
            e_draft_old = torch.zeros(384, device=self.device)
            q_old_draft = torch.tensor(0.0, device=self.device)

        comp_old = {}
        for aid in self.possible_agents:
            if _skip_r:
                dim = 384
                e_local = torch.zeros(dim, device=self.device)
                e_feedback = torch.zeros(dim, device=self.device)
                e_global = torch.zeros(dim, device=self.device)
            else:
                e_local = torch.from_numpy(embed(self.local_cache[aid]["memory"])).to(self.device)
                e_feedback = torch.from_numpy(embed(self.feedback)).to(self.device)
                e_global = torch.from_numpy(embed(self.global_knowledge)).to(self.device)
            q_L = self.get_q_torch(e_local, e_gold, "local", aid=aid, text_raw=self.local_cache[aid]["memory"])
            q_F = self.get_q_torch(e_feedback, e_gold, "feedback", extra_emb=e_draft_old, text_raw=self.feedback)
            q_G = self.get_q_torch(e_global, e_gold, "global", extra_emb=e_global, text_raw=self.global_knowledge)
            comp_old[aid] = {"L": q_L, "F": q_F, "G": q_G}

        # Store old global knowledge text for novelty calc
        old_global_knowledge = self.global_knowledge

        # 4. Execute Actions — parallel across agents (each LLM call is I/O-bound)
        norm_actions = {aid: (int(a["act"]) if isinstance(a, dict) else int(a)) for aid, a in actions.items()}
        self.list_actions.append(norm_actions)

        failed_agents = set()

        def _run_agent(aid: str, act: int):
            try:
                msg = ACTION_DISPATCHER[AgentAction(act)].execute(self, aid)
                return aid, msg, None
            except Exception as e:
                return aid, None, e

        with ThreadPoolExecutor(max_workers=len(norm_actions)) as pool:
            futures = {pool.submit(_run_agent, aid, act): aid
                       for aid, act in norm_actions.items()}
            for fut in as_completed(futures):
                aid, msg, err = fut.result()
                if err is not None:
                    print(f"[Error] {aid} failed: {err}")
                    failed_agents.add(aid)
                else:
                    print(f"    [Action] {msg}")

        # 5. Calculate New Quality & Rewards (skipped at inference)
        if _skip_r:
            e_draft_new = torch.zeros(384, device=self.device)
            e_global_old = torch.zeros(384, device=self.device)
        else:
            e_draft_new = torch.from_numpy(embed(self.draft)).to(self.device)
            e_global_old = torch.from_numpy(embed(old_global_knowledge)).to(self.device)
        q_new_draft = self.get_q_torch(e_draft_new, e_gold, "draft", text_raw=self.draft)
        draft_gain = q_new_draft - q_old_draft

        rewards = {}
        for aid in self.agents_list:
            if aid in failed_agents:
                rewards[aid] = -0.5
                continue

            if aid not in norm_actions:
                rewards[aid] = 0.0
                continue

            act_type = AgentAction(norm_actions[aid])

            if act_type == AgentAction.WRITE:
                if _skip_r:
                    rewards[aid] = 0.0
                else:
                    credit = 1.0 if aid == self.last_writer else self.write_credit_ratio
                    r_write = credit * float(draft_gain.item())
                    rewards[aid] = self.reward_norm[aid].update(r_write)
            else:
                if _skip_r:
                    q_new = torch.tensor(0.0, device=self.device)
                    q_old = torch.tensor(0.0, device=self.device)
                elif act_type == AgentAction.SEARCH:
                    e_new = torch.from_numpy(embed(self.local_cache[aid]["memory"])).to(self.device)
                    q_new = self.get_q_torch(e_new, e_gold, "local", aid=aid, text_raw=self.local_cache[aid]["memory"])
                    q_old = comp_old[aid]["L"]
                elif act_type == AgentAction.FEEDBACK:
                    e_new = torch.from_numpy(embed(self.feedback)).to(self.device)
                    q_new = self.get_q_torch(e_new, e_gold, "feedback", extra_emb=e_draft_new, text_raw=self.feedback)
                    q_old = comp_old[aid]["F"]
                elif act_type == AgentAction.UPDATE:
                    e_new = torch.from_numpy(embed(self.global_knowledge)).to(self.device)
                    q_new = self.get_q_torch(e_new, e_gold, "global", extra_emb=e_global_old, text_raw=self.global_knowledge)
                    q_old = comp_old[aid]["G"]
                else:
                    q_new = torch.tensor(0.0, device=self.device)
                    q_old = torch.tensor(0.0, device=self.device)

                comp_gain = q_new - q_old
                if _skip_r:
                    r_total = 0.0
                else:
                    r_total = float((self.draft_ambient_gamma * draft_gain + (1.0 - self.draft_ambient_gamma) * comp_gain).item())
                rewards[aid] = self.reward_norm[aid].update(r_total)

        self.round += 1
        return self.get_obs(), rewards, {aid: self.round >= self.max_rounds for aid in self.agents_list}, {aid: False for aid in self.agents_list}, {aid: {} for aid in self.agents_list}

    def build_action(self) -> Dict[str, np.ndarray]:
        counts = {a: np.zeros(NUM_ACTIONS, dtype=np.float32) for a in self.possible_agents}
        totals = {a: 0.0 for a in self.possible_agents}

        for step_actions in self.list_actions:
            for a, act in step_actions.items():
                if 0 <= int(act) < NUM_ACTIONS:
                    counts[a][int(act)] += 1.0
                    totals[a] += 1.0

        dist = {}
        for a in self.possible_agents:
            if totals[a] > 0:
                dist[a] = counts[a] / float(totals[a])
            else:
                dist[a] = counts[a]  # all zeros
        return dist

    def paper_ids(self,aid: str) -> set:
        ids = set()
        hist = self.read_history.get(aid, [])
        for item in hist:
            if isinstance(item, dict):
                if "id" in item:
                    ids.add(item["id"])
                else:
                    for k in item.keys():
                        ids.add(k)
            else:
                ids.add(item)
        return ids
    def render(self, mode: str = "human"): return None
    def close(self): return None

RWG_Enviroment = RWG_Environment
