# src/rl_env/rwg_env_no_feedback.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
import gymnasium.spaces as spaces
import json
import os
import sys
import numpy as np

from src.agent.model import WorkerLLM
from src.agent.action_no_feedback import AgentAction, ACTION_DISPATCHER, NUM_ACTIONS
from src.core import config
from src.core.utils import (
    embed, cos, clip01, to_text,
    safe_text, tokenize, length_words, compress_len,
    conditional_complexity, ncd,
    semantic_sim, semantic_dist01,
    token_distribution, kl_div, js_div,
    compression_density_reward, sigmoid_scalar,
    load_papers, load_graph_out, load_graph_in
)

class RewardNormalizer:
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        
    def update(self, x: float) -> float:
        delta = x - self.mean
        self.mean += (1 - self.alpha) * delta
        self.var = self.alpha * self.var + (1 - self.alpha) * (delta ** 2)
        std = np.sqrt(max(self.var, self.eps))
        return (x - self.mean) / std

class RWG_Environment_NoFeedback:
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_agents: int = 2, llm=None, use_gpu: bool = False, skip_embed: bool = False, skip_reward_embed: bool = False, papers=None, citation_adj_in=None, citation_adj_out=None):
        self.skip_embed = skip_embed
        self.skip_reward_embed = skip_reward_embed
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents_list = self.possible_agents[:]
        self.papers: Dict[int, Dict[str, Any]] = papers if papers is not None else load_papers()
        self.global_knowledge: str = ""
        self.draft: str = ""
        self.round: int = 0
        self.max_rounds: int = 20
        self.paper_title: str = ""
        self.abstract: str = ""
        self.token_total: int = 0
        self.agent_token_usage: Dict[str, int] = {aid: 0 for aid in self.possible_agents}
        self.action_token_usage: Dict[str, Dict[str, int]] = {
            aid: {"L": 0, "G": 0, "D": 0} for aid in self.possible_agents
        }
        self.read_history: Dict[str, List[Dict[int, str]]] = {aid: [] for aid in self.possible_agents}
        self.quality_history: List[Dict[str, float]] = []
        self.quality_metrics: List[Dict[str, float]] = []
        self.current_actions: Dict[str, int] = {}
        self.list_actions: List[Dict[str,int]] = []
        self.local_cache: Dict[str, Dict[str, Any]] = {aid: {"memory": "", "rationale": ""} for aid in self.possible_agents}   
        self.llm = llm if llm is not None else WorkerLLM()
        self.citation_adj_in: Dict[int, List[int]] = citation_adj_in if citation_adj_in is not None else load_graph_in()
        self.citation_adj_out: Dict[int, List[int]] = citation_adj_out if citation_adj_out is not None else load_graph_out()
        self.latest_candidates: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in self.possible_agents}
        self.search_top_k: int = getattr(config, "SEARCH_TOP_K", 5)
        self.wP, self.wG,  self.wL = 1.0, 0.4, 0.3
        self.lam: float = 0.001
        self.episode_stats: Dict[str, Any] = {"actions_taken": {a.name: 0 for a in AgentAction}}
        self.obs_space = NUM_ACTIONS
        self.act_space = spaces.Discrete(NUM_ACTIONS)
        self.draft_target = ""
        self.cited_paper : List[Dict[str,str]] = []
        self.alpha :float = 0.4
        self.eps = 1e-8
        self.last_writer: str = ""
        self.write_credit_ratio: float = 0.3
        self.draft_ambient_gamma: float = 0.1
        self.reward_norm = {aid: RewardNormalizer() for aid in self.possible_agents}

    def update_local_cache(self, aid: str, response: Dict[str, Any]):
        self.local_cache[aid]["memory"] = response.get("memory", "")
        self.local_cache[aid]["rationale"] = response.get("rationale", "")

    def update_draft(self, aid: str, response: Dict[str, Any]):
        new_content = response.get("related_work", "")
        if not new_content:
            for k, v in response.items():
                if isinstance(v, str) and len(v) > len(new_content):
                    new_content = v
        if new_content:
            if not isinstance(new_content, str):
                new_content = json.dumps(new_content, ensure_ascii=False)
            self.draft = new_content
            self.last_writer = aid
    
    def update_tokens(self, agent_id: str, usage: int):
        self.agent_token_usage[agent_id] += usage
        self.token_total += usage

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if options is None: options = {}
        self.agents_list = self.possible_agents[:]
        self.max_rounds = options.get("max_rounds", self.max_rounds)
        self.round = 0
        self.draft = ""
        self.global_knowledge = ""
        self.paper_title = options.get("paper_title", "")
        self.abstract = options.get("abstract", "")
        self.current_actions = {}
        self.local_cache = {aid: {"memory": "", "rationale": ""} for aid in self.possible_agents}
        self.read_history = {aid: [] for aid in self.possible_agents}
        self.latest_candidates = {aid: [] for aid in self.possible_agents}
        self.cited_paper = []
        self.last_writer = ""
        self.token_total = 0
        self.agent_token_usage = {aid: 0 for aid in self.possible_agents}
        self.action_token_usage = {aid: {"L": 0, "G": 0, "D": 0} for aid in self.possible_agents}
        self.quality_history = []
        self.episode_stats = {"actions_taken": {a.name: 0 for a in AgentAction}}
        self.list_actions = []
        return self.get_obs(), {aid: {} for aid in self.agents_list}
    
    def get_obs(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        t_norm = clip01(self.round / self.max_rounds)
        total_counts = {a: 0 for a in range(NUM_ACTIONS)}
        agent_counts = {aid: {a: 0 for a in range(NUM_ACTIONS)} for aid in self.possible_agents}
        
        for turn_actions in self.list_actions:
            for aid, act in turn_actions.items():
                if aid in agent_counts:
                    total_counts[act] += 1
                    agent_counts[aid][act] += 1

        L_text = {a: to_text(self.local_cache.get(a, {}).get("memory", "")) for a in self.possible_agents}
        G_text = self.global_knowledge or ""
        D_text = self.draft or ""
        T_text = (self.paper_title + " " + (self.abstract or "")) if self.paper_title else (self.draft_target or "")

        if self.skip_embed:
            dim = 384
            eT = np.zeros(dim, dtype=np.float32)
            eG = np.zeros(dim, dtype=np.float32)
            eD = np.zeros(dim, dtype=np.float32)
            eL = {a: np.zeros(dim, dtype=np.float32) for a in self.possible_agents}
        else:
            eT = embed(T_text)
            eG = embed(G_text)
            eD = embed(D_text)
            eL = {a: embed(txt) for a, txt in L_text.items()}

        dist = self.build_action()

        def norm_tokens(t: int, budget: float = 2000.0) -> float:
            """Linear token-budget countdown: 1.0 (unused) → 0.0 (exhausted)."""
            return max(0.0, 1.0 - float(t) / budget)

        def safe_clip_social(total: int, mine: int) -> float:
            if total == 0: return 0.0
            return clip01(float((total - mine) / total))

        relD_global = clip01(float(cos(eD, eT)))

        for aid in self.agents_list:
            # SEARCH (0)
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

            # UPDATE (1)
            relG_agent = clip01(float(cos(eL[aid], eT)))
            novG_agent = clip01(1.0 - float(cos(eL[aid], eG)))
            impactG = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.UPDATE)]))
            tokG = norm_tokens(self.action_token_usage[aid]["G"])
            socialG = safe_clip_social(total_counts[1], agent_counts[aid][1])

            # WRITE (2)
            relD_row = relD_global
            covD = clip01(float(cos(eD, eG)))
            impactD = clip01(float(dist.get(aid, np.zeros(NUM_ACTIONS))[int(AgentAction.WRITE)]))
            tokD = norm_tokens(self.action_token_usage[aid]["D"])
            socialD = safe_clip_social(total_counts[2], agent_counts[aid][2])

            # Matrix: [Rel, Nov/Cov, Impact, tok_norm, social]  shape (3, 5)
            mat = np.array([
                [relL,       covL,       impactL, tokL, socialL], # SEARCH
                [relG_agent, novG_agent, impactG, tokG, socialG], # UPDATE
                [relD_row,   covD,       impactD, tokD, socialD], # WRITE
            ], dtype=np.float32)
            obs[aid] = mat
        return obs
    
    def safe_cos(self, a, b):
        a = a.view(1, -1)
        b = b.view(1, -1)
        if a.norm() < self.eps or b.norm() < self.eps:
            return torch.tensor(0.0, device=self.device)
        return F.cosine_similarity(a, b).squeeze()

    def get_q_torch(self, text_emb, target_emb, doc_type="draft", extra_emb=None, aid=None, text_raw=""):
        if text_emb is None: return torch.tensor(0.0, device=self.device)
        text_emb = text_emb.view(1, -1)
        target_emb = target_emb.view(1, -1)
        rel = self.safe_cos(text_emb, target_emb)

        if doc_type == "draft":
            struct_reward = compression_density_reward(text_raw)
            return 0.8 * rel + 0.2 * struct_reward
        elif doc_type == "local":
            topic_str = self.paper_title + " " + self.abstract
            topic_emb = torch.from_numpy(embed(topic_str[:1000])).to(self.device).view(1, -1)
            density = self.safe_cos(text_emb, topic_emb) 
            penalty = torch.tensor(0.0, device=self.device)
            if aid in self.read_history and self.read_history[aid]:
                hist_list = [to_text(h) for h in self.read_history[aid][-5:]]
                if hist_list:
                    history_embs = torch.stack([torch.from_numpy(embed(h)).to(self.device) for h in hist_list])
                    sims = [self.safe_cos(text_emb, history_embs[i:i+1]) for i in range(history_embs.size(0))]
                    if sims: penalty = torch.max(torch.stack(sims))
            return 0.5 * rel + 0.3 * density - 0.2 * penalty
        elif doc_type == "global":
            if extra_emb is not None:
                novelty = 1.0 - self.safe_cos(text_emb, extra_emb.view(1, -1))
            else:
                novelty = torch.tensor(0.0, device=self.device)
            p_dist = token_distribution(text_raw)
            gold_dist = token_distribution(self.draft_target)
            diversity = 1.0 - clip01(js_div(p_dist, gold_dist) / np.log(2))
            return 0.5 * rel + 0.3 * novelty + 0.2 * diversity
        return torch.tensor(0.0, device=self.device)

    def step(self, actions: Dict[str, Any]):
        _skip_r = self.skip_embed or self.skip_reward_embed

        # Reward-only embeddings — skipped at inference
        if not _skip_r:
            e_gold = torch.from_numpy(embed(self.draft_target)).to(self.device)
            e_draft_old = torch.from_numpy(embed(self.draft)).to(self.device)
            q_old_draft = self.get_q_torch(e_draft_old, e_gold, "draft", text_raw=self.draft)
        else:
            e_gold = torch.zeros(384, device=self.device)
            e_draft_old = torch.zeros(384, device=self.device)
            q_old_draft = torch.tensor(0.0, device=self.device)

        comp_old = {}
        for aid in self.possible_agents:
            if _skip_r:
                e_local = torch.zeros(384, device=self.device)
                e_global = torch.zeros(384, device=self.device)
            else:
                e_local = torch.from_numpy(embed(self.local_cache[aid]["memory"])).to(self.device)
                e_global = torch.from_numpy(embed(self.global_knowledge)).to(self.device)
            q_L = self.get_q_torch(e_local, e_gold, "local", aid=aid, text_raw=self.local_cache[aid]["memory"])
            q_G = self.get_q_torch(e_global, e_gold, "global", extra_emb=e_global, text_raw=self.global_knowledge)
            comp_old[aid] = {"L": q_L, "G": q_G}

        old_global_knowledge = self.global_knowledge
        norm_actions = {aid: (int(a["act"]) if isinstance(a, dict) else int(a)) for aid, a in actions.items()}
        self.list_actions.append(norm_actions)

        failed_agents = set()
        for aid, act in norm_actions.items():
            try:
                ACTION_DISPATCHER[AgentAction(act)].execute(self, aid)
            except Exception as e:
                failed_agents.add(aid)

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
                    q_new, q_old = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                elif act_type == AgentAction.SEARCH:
                    e_new = torch.from_numpy(embed(self.local_cache[aid]["memory"])).to(self.device)
                    q_new = self.get_q_torch(e_new, e_gold, "local", aid=aid, text_raw=self.local_cache[aid]["memory"])
                    q_old = comp_old[aid]["L"]
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
            dist[a] = counts[a] / totals[a] if totals[a] > 0 else counts[a]
        return dist

    def paper_ids(self,aid: str) -> set:
        ids = set()
        hist = self.read_history.get(aid, [])
        for item in hist:
            if isinstance(item, dict):
                ids.add(item.get("id"))
            else:
                ids.add(item)
        return ids
    def render(self, mode: str = "human"): return None
    def close(self): return None
