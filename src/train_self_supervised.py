#!/usr/bin/env python3
# src/train_self_supervised.py
"""
Self-Supervised Training for MARS-RWG

Training using papers from utils database + citation graph:
- Input: title + abstract
- Gold standard: related_work section
- Reward: similarity(generated, gold_standard) + optional shaped env reward

This version:
- Reuses existing utils functions:
  - load_papers()
  - load_graph_in()
  - load_graph_out()
  - EmbeddingEncoder
- Trains only on papers with:
  - non-empty title / abstract / related_work
  - enough valid citation neighbors
- Uses test_env-style setup for SEARCH:
  - papers[-1] = target paper
  - graph_out[-1] = target citations
  - read_history starts from -1
- Adds clearer training logs:
  - selected paper count
  - env steps / epoch / total
  - agent steps / epoch / total
  - per-paper sim / reward
  - actor loss / critic loss / entropy
  - global env step progress

Usage:
    python src/train_self_supervised.py
    python src/train_self_supervised.py -p 10 -s 15
    python src/train_self_supervised.py --scratch
    python src/train_self_supervised.py --cold-start
"""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rl_env.rwg_env import RWG_Enviroment
from src.training.ippo_trainer import RolloutBuffer
from src.networks.actor import Actor
from src.networks.critic import Critic
from src.agent.model import WorkerLLM
from src.core.utils import EmbeddingEncoder, load_papers, load_graph_in, load_graph_out
from src.core.tracker import ActionTracker


ACTION_NAMES = {0: "SEARCH", 1: "UPDATE", 2: "WRITE", 3: "FEEDBACK"}
HEURISTIC_STEPS = 4
DEFAULT_CKPT_DIR = "./checkpoints/self_supervised"


def plot_cumulative_progress(scores: List[float], save_path: str):
    if not scores:
        return

    cum_scores = np.cumsum(scores)
    indices = np.arange(1, len(scores) + 1)
    avgs = cum_scores / indices

    plt.figure(figsize=(10, 6))
    plt.plot(avgs, marker="", linestyle="-", linewidth=2, label="Cumulative Avg")
    plt.plot(scores, alpha=0.2, linewidth=1, label="Raw Score")

    plt.title(f"Training Progress (Cumulative Avg) - {len(scores)} Samples", fontsize=14, fontweight="bold")
    plt.xlabel("Total Papers Processed", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


def plot_epoch_violin(epoch_data: List[List[float]], save_path: str):
    if not epoch_data:
        return

    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(epoch_data, showmeans=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    plt.title("Score Distribution per Epoch (Violin Plot)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xticks(range(1, len(epoch_data) + 1), [str(i) for i in range(1, len(epoch_data) + 1)])
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    except Exception as e:
        print(f"Error saving violin plot: {e}")
    plt.close()


def compute_similarity(gen: str, gold: str, embedder: EmbeddingEncoder) -> float:
    if not gen.strip() or not gold.strip():
        return 0.0
    try:
        g_emb = embedder.encode([gen])[0]
        gold_emb = embedder.encode([gold])[0]
        sim = np.dot(g_emb, gold_emb) / (np.linalg.norm(g_emb) * np.linalg.norm(gold_emb) + 1e-8)
        return max(0.0, min(1.0, float(sim)))
    except Exception:
        return 0.0


def compute_training_stats(num_papers: int, steps: int, epochs: int, agents: int) -> Dict[str, int]:
    env_steps_per_epoch = num_papers * steps
    agent_steps_per_epoch = num_papers * steps * agents
    total_env_steps = env_steps_per_epoch * epochs
    total_agent_steps = agent_steps_per_epoch * epochs
    return {
        "env_steps_per_epoch": env_steps_per_epoch,
        "agent_steps_per_epoch": agent_steps_per_epoch,
        "total_env_steps": total_env_steps,
        "total_agent_steps": total_agent_steps,
    }


def build_cited_paper(seed_id: int, papers: Dict[int, Dict[str, Any]], graph_out: Dict[int, List[int]], max_refs: int = 10):
    cited_ids = (graph_out.get(seed_id, []) or [])[:max_refs]
    out = []
    for pid in cited_ids:
        if pid in papers:
            p = papers[pid]
            out.append(
                {
                    "id": pid,
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "structure": p.get("structure", []),
                }
            )
    return out


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = Path(ckpt_dir)
    if not path.exists():
        return None

    if (path / "selected").exists():
        print("   Using 'selected' checkpoint")
        return str(path / "selected")

    ckpts = [d for d in path.iterdir() if d.is_dir() and d.name != "selected"]
    if not ckpts:
        return None

    ckpts = sorted(ckpts, key=lambda x: x.name, reverse=True)
    return str(ckpts[0])


def get_best_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = Path(ckpt_dir)
    if not path.exists():
        return None

    if (path / "selected").exists():
        print("   Using 'selected' checkpoint")
        return str(path / "selected")

    ckpts = [d for d in path.iterdir() if d.is_dir() and d.name != "selected"]
    if not ckpts:
        return None

    best_ckpt = None
    best_sim = -1.0

    for c in ckpts:
        stats_f = c / "stats.pth"
        if stats_f.exists():
            try:
                stats = torch.load(stats_f, map_location="cpu")
                sim = float(stats.get("best_sim", 0.0))
                if sim > best_sim:
                    best_sim = sim
                    best_ckpt = c
            except Exception:
                pass

    if best_ckpt:
        print(f"   Using best checkpoint (sim={best_sim:.3f})")
        return str(best_ckpt)

    ckpts = sorted(ckpts, key=lambda x: x.name, reverse=True)
    return str(ckpts[0]) if ckpts else None


class Trainer:
    """Self-supervised trainer for MARS-RWG using graph-backed papers from utils."""

    def __init__(
        self,
        agents: int = 3,
        epochs: int = 3,
        steps: int = 15,
        paper_percent: float = 100.0,
        cold_start: bool = False,
        ckpt_dir: str = DEFAULT_CKPT_DIR,
        from_scratch: bool = False,
        ckpt_path: str = None,
        use_best: bool = False,
        use_wandb: bool = True,
        model_name: str = None,
        limit: int = None,
        min_neighbors: int = 5,
        save_freq: int = 1,
        paper_save_freq: int = 6,
    ):
        self.agents = agents
        self.epochs = epochs
        self.steps = steps
        self.paper_percent = paper_percent
        self.cold_start = cold_start
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.limit = limit
        self.min_neighbors = min_neighbors
        self.save_freq = save_freq
        self.paper_save_freq = paper_save_freq
        self.total_papers_processed = 0

        print("   Tell Trainer to load EmbeddingEncoder...")
        self.embedder = EmbeddingEncoder()

        print("   Loading papers_db + citation graphs from utils...")
        self.papers_db: Dict[int, Dict[str, Any]] = load_papers()
        self.graph_in: Dict[int, List[int]] = load_graph_in()
        self.graph_out: Dict[int, List[int]] = load_graph_out()

        if not self.papers_db:
            raise RuntimeError("papers_db is empty. Check src/core/utils.py load_papers().")

        self.paper_ids = self._build_training_ids(
            papers_db=self.papers_db,
            graph_out=self.graph_out,
            paper_percent=paper_percent,
            limit=limit,
            min_neighbors=min_neighbors,
        )

        self.training_stats = compute_training_stats(
            num_papers=len(self.paper_ids),
            steps=self.steps,
            epochs=self.epochs,
            agents=self.agents,
        )
        self.global_env_step = 0
        self.global_agent_step = 0
        self.total_tokens = 0 # Global token counter

        print(f"   Loaded {len(self.paper_ids)} papers for training")

        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(
                project="mars-rwg",
                name=f"train_{datetime.now().strftime('%m%d_%H%M')}",
                config={
                    "agents": agents,
                    "epochs": epochs,
                    "steps": steps,
                    "paper_percent": paper_percent,
                    "selected_papers": len(self.paper_ids),
                    "cold_start": cold_start,
                    "model": model_name or "openrouter:qwen/qwen-2.5-72b-instruct",
                    "min_neighbors": min_neighbors,
                    "env_steps_per_epoch": self.training_stats["env_steps_per_epoch"],
                    "agent_steps_per_epoch": self.training_stats["agent_steps_per_epoch"],
                    "total_env_steps": self.training_stats["total_env_steps"],
                    "total_agent_steps": self.training_stats["total_agent_steps"],
                },
            )

        print("\n   Initializing...")
        llm = WorkerLLM(model_name=model_name) if model_name else WorkerLLM()
        self.env = RWG_Enviroment(
            num_agents=agents, llm=llm, use_gpu=torch.cuda.is_available(),
            papers=self.papers_db,
            citation_adj_in=self.graph_in,
            citation_adj_out=self.graph_out,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_networks()

        self.loaded_ckpt = None
        self.best_sim = 0.0
        if not from_scratch:
            if ckpt_path:
                ckpt = ckpt_path
            elif use_best:
                ckpt = get_best_checkpoint(str(self.ckpt_dir))
            else:
                ckpt = get_latest_checkpoint(str(self.ckpt_dir))

            if ckpt:
                self._load_checkpoint(ckpt)
                self.loaded_ckpt = ckpt

        self.buffer = RolloutBuffer(self.env.possible_agents)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_clip = 0.2
        self.k_epochs = 4
        self.entropy_coeff = 0.01

        self.log_dir = os.path.join(self.ckpt_dir, "logs")
        self.tracker = ActionTracker(self.env.possible_agents, log_dir=self.log_dir)

    def _build_training_ids(
        self,
        papers_db: Dict[int, Dict[str, Any]],
        graph_out: Dict[int, List[int]],
        paper_percent: float,
        limit: Optional[int],
        min_neighbors: int,
    ) -> List[int]:
        all_ids = list(papers_db.keys())
        np.random.shuffle(all_ids)

        valid_ids: List[int] = []
        for pid in all_ids:
            p = papers_db[pid]
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            rw = (p.get("related_work") or "").strip()

            if len(title) <= 10 or len(abstract) <= 50 or len(rw) <= 100:
                continue

            neigh = [x for x in (graph_out.get(pid, []) or []) if x in papers_db]
            if len(neigh) >= min_neighbors:
                valid_ids.append(pid)

        if limit is not None:
            return valid_ids[:limit]

        if paper_percent < 100.0:
            n = max(1, int(len(valid_ids) * paper_percent / 100.0))
            valid_ids = valid_ids[:n]

        return valid_ids

    def _init_networks(self):
        # Matrix 4x5 (Relevance, Coverage, Impact, Tokens, Social) = 20
        obs_dim = 20
        act_dim = 4

        # IPPO with Parameter Sharing: Single shared network for all agents
        self.shared_actor = Actor(obs_dim, act_dim).to(self.device)
        self.shared_critic = Critic(obs_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.shared_actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.shared_critic.parameters(), lr=1e-3)

        print(f"   [OK] Shared Networks (Parameter Sharing): obs={obs_dim}, act={act_dim}, hidden=32")

    def _get_heuristic_action(self, aid: str) -> int:
        local_cache = self.env.local_cache.get(aid, {})
        mem = local_cache.get("memory", "")
        if not str(mem).strip():
            return 0  # Step 0: SEARCH

        glob = self.env.global_knowledge or ""
        if not glob.strip():
            return 1  # Step 1: UPDATE

        feedback = self.env.feedback or ""
        if not feedback.strip():
            return 3  # Step 2: FEEDBACK

        return 2  # Step 3: WRITE

    def _load_checkpoint(self, path: str):
        p = Path(path)
        if not p.exists():
            print(f"   [WARN] Checkpoint not found: {path}")
            return

        # Auto-detect subdirectories
        if not (p / "actor.pth").exists():
            for sub in ["latest", "selected"]:
                if (p / sub / "actor.pth").exists():
                    p = p / sub
                    print(f"   [INFO] Auto-detected checkpoint subdirectory: {sub}")
                    break

        print(f"   [INFO] Loading shared checkpoint: {p}")
        actor_f = p / "actor.pth"
        critic_f = p / "critic.pth"
        
        # Compatibility check: if old per-agent files exist, load first one as shared
        if not actor_f.exists() and (p / f"actor_{self.env.possible_agents[0]}.pth").exists():
             actor_f = p / f"actor_{self.env.possible_agents[0]}.pth"
             critic_f = p / f"critic_{self.env.possible_agents[0]}.pth"

        if actor_f.exists():
            self.shared_actor.load_state_dict(torch.load(actor_f, map_location=self.device))
            print(f"   [OK] Loaded Actor weights.")
        else:
            print(f"   [ERROR] Could not find actor.pth in {p}")

        if critic_f.exists():
            self.shared_critic.load_state_dict(torch.load(critic_f, map_location=self.device))
            print(f"   [OK] Loaded Critic weights.")

        stats_f = p / "stats.pth"
        if stats_f.exists():
            stats = torch.load(stats_f, map_location=self.device)
            self.best_sim = float(stats.get("best_sim", 0.0))
            print(f"   [INFO] Resumed from epoch {stats.get('epoch', '?')}, best_sim={self.best_sim:.3f}")

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            arr = obs.flatten().astype(np.float32)
            if arr.size < 20:
                arr = np.pad(arr, (0, 20 - arr.size))
            return arr[:20]

        if isinstance(obs, dict):
            # Fallback for dict-based obs if ever used in training
            x = np.array(obs.get("x_features", np.zeros(15)), dtype=np.float32).flatten()[:15]
            s = np.array(obs.get("social", np.zeros(5)), dtype=np.float32).flatten()[:5]
            arr = np.concatenate([x, s])
            if arr.size < 20:
                arr = np.pad(arr, (0, 20 - arr.size))
            return arr[:20].astype(np.float32)

        return np.zeros(20, dtype=np.float32)

    def _select_action(self, obs: Any, aid: str) -> Tuple[int, float, float]:
        obs_t = torch.tensor(self._flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.shared_actor(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            a = int(action.item())
            if a >= 4:
                a = 0
            logp = float(dist.log_prob(torch.tensor(a, device=self.device)).item())
            val = float(self.shared_critic(obs_t).item())
            return a, logp, val

    def _update_policy(self) -> Dict[str, float]:
        advantages, returns = self.buffer.compute_returns_advantages(self.gamma, self.gae_lambda, self.device)

        # Aggregate data from all agents for Parameter Sharing update
        all_obs = []
        all_actions = []
        all_old_lp = []
        all_adv = []
        all_ret = []

        for aid in self.env.possible_agents:
            obs_data = self.buffer.data[aid]["obs"]
            if not obs_data:
                continue

            all_obs.append(np.array(obs_data))
            all_actions.append(np.array(self.buffer.data[aid]["actions"]))
            all_old_lp.append(np.array(self.buffer.data[aid]["logprobs"]))
            
            # Normalize advantages per agent
            adv = advantages[aid]
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            all_adv.append(adv.cpu().numpy())
            all_ret.append(returns[aid].cpu().numpy())

        if not all_obs:
            self.buffer.reset()
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

        # Convert to big tensors
        obs = torch.tensor(np.concatenate(all_obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.concatenate(all_actions), dtype=torch.int64, device=self.device)
        old_lp = torch.tensor(np.concatenate(all_old_lp), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.concatenate(all_adv), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.concatenate(all_ret), dtype=torch.float32, device=self.device)

        actor_losses = []
        critic_losses = []
        entropies = []

        # Perform PPO updates on shared model
        for _ in range(self.k_epochs):
            logits = self.shared_actor(obs)
            dist = Categorical(logits=logits)
            lp = dist.log_prob(actions)
            ent = dist.entropy().mean()
            vals = self.shared_critic(obs).squeeze(-1)

            ratio = (lp - old_lp).exp()
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
            
            actor_loss = -torch.min(s1, s2).mean()
            critic_loss = torch.nn.functional.mse_loss(vals, ret)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * ent

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))
            entropies.append(float(ent.item()))

        self.buffer.reset()

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def train(self):
        print("\n" + "=" * 70)
        print("  TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"   Epochs:                {self.epochs}")
        print(f"   Steps/paper:           {self.steps}")
        print(f"   Agents:                {self.agents}")
        print(f"   Cold start:            {self.cold_start}")
        print(f"   Min neighbors:         {self.min_neighbors}")
        print(f"   Checkpoint:            {self.loaded_ckpt or 'None (from scratch)'}")
        print()
        print(f"   Total DB papers:       {len(self.papers_db)}")
        print(f"   Selected train papers: {len(self.paper_ids)}")
        print(f"   Paper percent:         {self.paper_percent}%")
        print()
        print(f"   Env steps / epoch:     {self.training_stats['env_steps_per_epoch']}")
        print(f"   Agent steps / epoch:   {self.training_stats['agent_steps_per_epoch']}")
        print(f"   Total env steps:       {self.training_stats['total_env_steps']}")
        print(f"   Total agent steps:     {self.training_stats['total_agent_steps']}")
        print("=" * 70)
        print("\n[INFO] Sample papers:")
        for i, pid in enumerate(self.paper_ids[:5]):
            p = self.papers_db[pid]
            title = p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
            print(f"   {i+1}. [{pid}] {title}")
        if len(self.paper_ids) > 5:
            print(f"   ... and {len(self.paper_ids) - 5} more")
        print("=" * 70 + "\n")

        self.run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        epoch_pbar = tqdm(
            range(self.epochs),
            desc="  Epochs",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        all_scores_flat: List[float] = []
        epoch_scores_list: List[List[float]] = []

        for epoch in epoch_pbar:
            np.random.shuffle(self.paper_ids)
            sims: List[float] = []
            epoch_rewards: List[float] = []
            epoch_actor_losses: List[float] = []
            epoch_critic_losses: List[float] = []
            epoch_entropies: List[float] = []

            paper_pbar = tqdm(
                self.paper_ids,
                desc=f"   Epoch {epoch+1}",
                position=1,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )

            for pid in paper_pbar:
                paper_info = self._train_paper(pid)
                sim = paper_info["final_sim"]
                sims.append(sim)
                all_scores_flat.append(sim)
                epoch_rewards.append(paper_info["avg_step_reward"])

                loss_info = self._update_policy()
                epoch_actor_losses.append(loss_info["actor_loss"])
                epoch_critic_losses.append(loss_info["critic_loss"])
                epoch_entropies.append(loss_info["entropy"])

                self._finalize_paper_tokens()

                self.total_papers_processed += 1
                if self.paper_save_freq > 0 and self.total_papers_processed % self.paper_save_freq == 0:
                    self._save_checkpoint_paper(self.total_papers_processed)

                if self.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "sim": sim,
                            "avg_sim": float(np.mean(sims)),
                            "avg_step_reward": paper_info["avg_step_reward"],
                            "avg_step_sim": paper_info["avg_step_sim"],
                            "draft_words": paper_info["draft_words"],
                            "neighbors": paper_info["neighbors"],
                            "actor_loss": loss_info["actor_loss"],
                            "critic_loss": loss_info["critic_loss"],
                            "entropy": loss_info["entropy"],
                            "env_step": self.global_env_step,
                            "agent_step": self.global_agent_step,
                            "total_tokens": self.total_tokens + self.env.token_total,
                        }
                    )

                paper_pbar.set_postfix_str(
                    f"sim={sim:.3f} | avg={np.mean(sims):.3f} | "
                    f"r={paper_info['avg_step_reward']:.3f} | "
                    f"aloss={loss_info['actor_loss']:.3f} | "
                    f"closs={loss_info['critic_loss']:.3f}"
                )

                plot_path = os.path.join("outputs", "training_progress_cumulative.png")
                plot_cumulative_progress(all_scores_flat, plot_path)

            paper_pbar.close()

            epoch_scores_list.append(sims)
            violin_path = os.path.join("outputs", "training_violin_epochs.png")
            plot_epoch_violin(epoch_scores_list, violin_path)

            avg_sim = float(np.mean(sims)) if sims else 0.0
            avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            avg_actor_loss = float(np.mean(epoch_actor_losses)) if epoch_actor_losses else 0.0
            avg_critic_loss = float(np.mean(epoch_critic_losses)) if epoch_critic_losses else 0.0
            avg_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0

            epoch_pbar.set_postfix_str(f"avg_sim={avg_sim:.3f}")

            if avg_sim > self.best_sim:
                self.best_sim = avg_sim
            
            # Save checkpoint every N epochs (self.save_freq)
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch=epoch)
            else:
                # Always update 'latest' to allow resumption, but don't create new epoch folder
                self._save_checkpoint(epoch=None)

            tqdm.write(
                f"   [OK] Epoch {epoch+1} complete | "
                f"avg_sim={avg_sim:.3f} | avg_reward={avg_reward:.3f} | "
                f"actor_loss={avg_actor_loss:.3f} | critic_loss={avg_critic_loss:.3f} | "
                f"entropy={avg_entropy:.3f} | best={self.best_sim:.3f} | "
                f"total_tokens={self.total_tokens + self.env.token_total}"
            )

        epoch_pbar.close()
        print("\n  TRAINING COMPLETE!")
        print(f"   Best similarity: {self.best_sim:.3f}")
        print(f"   Final env step:  {self.global_env_step}")
        print(f"   Final agent step:{self.global_agent_step}")
        print(f"   Checkpoint: {self.ckpt_dir / self.run_timestamp}")

        if self.use_wandb and wandb is not None:
            wandb.finish()

    def _train_paper(self, pid: int) -> Dict[str, Any]:
        paper = self.papers_db[pid]
        gold = (paper.get("related_work") or "").strip()
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        real_citations = self.graph_out.get(pid, []) or []
        valid_neighbors = [x for x in real_citations if x in self.papers_db]

        # Add special -1 entry directly on the shared dict (sequential, no concurrency risk)
        self.papers_db[-1] = paper
        self.graph_out[-1] = valid_neighbors

        self.env.papers = self.papers_db
        self.env.citation_adj_out = self.graph_out
        self.env.citation_adj_in = self.graph_in
        self.env.draft_target = gold

        obs, _ = self.env.reset(
            options={
                "prompt": f"Write related work for: {title}",
                "paper_title": title,
                "abstract": abstract,
                "max_rounds": self.steps,
            },
            clear_memory=True,
        )
        self.env.list_actions = []  # env.reset() does not clear this; fix action-count accumulation

        for aid in self.env.possible_agents:
            self.env.read_history[aid] = [{"id": -1}]

        try:
            self.env.cited_paper = build_cited_paper(pid, self.papers_db, self.graph_out, max_refs=10)
        except Exception:
            pass

        step_rewards = []
        step_sims = []

        for step in range(self.steps):
            current_actions = {}
            actions, lps, vals = {}, {}, {}

            for aid in self.env.possible_agents:
                if self.cold_start and step < HEURISTIC_STEPS:
                    act = self._get_heuristic_action(aid)
                    obs_t = torch.tensor(self._flatten_obs(obs[aid]), dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.shared_actor(obs_t)
                        dist = Categorical(logits=logits)
                        lps[aid] = float(dist.log_prob(torch.tensor([act], device=self.device)).item())
                        vals[aid] = float(self.shared_critic(obs_t).item())
                else:
                    act, lp, val = self._select_action(obs[aid], aid)
                    lps[aid] = lp
                    vals[aid] = val

                actions[aid] = act
                current_actions[aid] = act

            next_obs, rewards, terms, truncs, _ = self.env.step(actions)

            self.global_env_step += 1
            self.global_agent_step += len(self.env.possible_agents)

            draft = self.env.draft or ""
            sim = compute_similarity(draft, gold, self.embedder)

            reward_mean = float(np.mean(list(rewards.values()))) if rewards else 0.0
            step_rewards.append(reward_mean)
            step_sims.append(sim)

            # Detailed logging per agent
            rewards_str = ", ".join([f"{aid}: {r:.4f}" for aid, r in rewards.items()])
            action_names = {aid: ACTION_NAMES[a] for aid, a in current_actions.items()}
            
            print(f"\n" + "-" * 60)
            print(f" [STEP] {step+1}/{self.steps} | Paper: {pid} | global_step: {self.global_env_step}")
            print(f" Title   : {title[:70]}...")
            print(f" Actions : {action_names}")
            print(f" Rewards : {rewards_str} (mean: {reward_mean:.4f})")
            print(f" Sim     : {sim:.4f} | Words: {len(draft.split())} (Draft), {len((self.env.global_knowledge or '').split())} (Global)")
            print("-" * 60)

            self.tracker.log_step(step, current_actions, env=self.env, rewards=rewards)

            for aid in self.env.possible_agents:
                done = terms.get(aid, False) or truncs.get(aid, False)
                shaped_reward = sim + rewards.get(aid, 0.0) * 0.1
                self.buffer.add(
                    aid,
                    self._flatten_obs(obs[aid]),
                    actions[aid],
                    lps[aid],
                    shaped_reward,
                    done,
                    vals[aid],
                )

            obs = next_obs
            if all(terms.values()):
                break

        for aid in self.env.possible_agents:
            if aid in obs:
                obs_t = torch.tensor(self._flatten_obs(obs[aid]), dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    self.buffer.data[aid]["values"].append(float(self.shared_critic(obs_t).item()))
            else:
                self.buffer.data[aid]["values"].append(0.0)

        final_draft = self.env.draft or ""
        final_sim = compute_similarity(final_draft, gold, self.embedder)

        return {
            "pid": pid,
            "title": title,
            "final_sim": final_sim,
            "avg_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "avg_step_sim": float(np.mean(step_sims)) if step_sims else 0.0,
            "neighbors": len(valid_neighbors),
            "draft_words": len(final_draft.split()),
        }
    
    def _finalize_paper_tokens(self):
        """Add current env tokens to the global counter."""
        self.total_tokens += self.env.token_total

    def _save_checkpoint(self, epoch: int = None):
        base_path = self.ckpt_dir / self.run_timestamp
        p = base_path / (f"epoch_{epoch + 1}" if epoch is not None else "latest")
        p.mkdir(parents=True, exist_ok=True)

        torch.save(self.shared_actor.state_dict(), p / "actor.pth")
        torch.save(self.shared_critic.state_dict(), p / "critic.pth")

        torch.save(
            {
                "timestamp": self.run_timestamp,
                "best_sim": self.best_sim,
                "epoch": epoch if epoch is not None else -1,
                "global_env_step": self.global_env_step,
                "global_agent_step": self.global_agent_step,
                "token_total": self.total_tokens,
                "action_token_usage": self.env.action_token_usage,
            },
            p / "stats.pth",
        )

        # Also save a human-readable token log
        try:
            with open(base_path / "token_usage.json", "w") as f:
                import json
                json.dump({
                    "total": self.total_tokens,
                    "per_agent": self.env.agent_token_usage,
                    "per_action": self.env.action_token_usage,
                    "estimated_cost_usd": (self.total_tokens / 1_000_000) * 0.15 
                }, f, indent=4)
        except Exception:
            pass

        if epoch is not None:
            latest_p = base_path / "latest"
            latest_p.mkdir(parents=True, exist_ok=True)
            torch.save(self.shared_actor.state_dict(), latest_p / "actor.pth")
            torch.save(self.shared_critic.state_dict(), latest_p / "critic.pth")
            torch.save(
                {
                    "timestamp": self.run_timestamp,
                    "best_sim": self.best_sim,
                    "epoch": epoch,
                    "global_env_step": self.global_env_step,
                    "global_agent_step": self.global_agent_step,
                },
                latest_p / "stats.pth",
            )

        print(f"   [SAVE] Saved checkpoint to: {p}")

    def _save_checkpoint_paper(self, paper_count: int):
        """Save checkpoint keyed by total papers processed (for fine-grained tracking)."""
        base_path = self.ckpt_dir / self.run_timestamp
        p = base_path / f"paper_{paper_count}"
        p.mkdir(parents=True, exist_ok=True)

        torch.save(self.shared_actor.state_dict(), p / "actor.pth")
        torch.save(self.shared_critic.state_dict(), p / "critic.pth")
        torch.save(
            {
                "timestamp": self.run_timestamp,
                "best_sim": self.best_sim,
                "total_papers": paper_count,
                "global_env_step": self.global_env_step,
                "global_agent_step": self.global_agent_step,
            },
            p / "stats.pth",
        )
        print(f"   [SAVE] Paper checkpoint: {p}")


def main():
    parser = argparse.ArgumentParser(description="Train MARS-RWG (self-supervised, graph-backed)")
    parser.add_argument("-a", type=int, default=3, help="Agents (default: 3)")
    parser.add_argument("-e", type=int, default=3, help="Epochs (default: 3)")
    parser.add_argument("-s", type=int, default=15, help="Steps per paper (default: 15)")
    parser.add_argument("-p", type=float, default=100, help="Paper percent (default: 100)")
    parser.add_argument("--cold-start", action="store_true", help="Force cold start sequence")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (random weights)")
    parser.add_argument("--best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from specific checkpoint path")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR, help="Checkpoint dir")
    parser.add_argument("--model", type=str, default=None, help="LLM model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to train on")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Min valid citation neighbors")
    parser.add_argument("--save-freq", type=int, default=1, help="Save frequency (epochs)")
    parser.add_argument("--paper-save-freq", type=int, default=6, help="Save checkpoint every N papers (default: 6)")

    args = parser.parse_args()

    Trainer(
        agents=args.a,
        epochs=args.e,
        steps=args.s,
        paper_percent=args.p,
        cold_start=args.cold_start,
        ckpt_dir=args.ckpt_dir,
        from_scratch=args.scratch,
        ckpt_path=args.ckpt,
        use_best=args.best,
        use_wandb=not args.no_wandb,
        model_name=args.model,
        limit=args.limit,
        min_neighbors=args.min_neighbors,
        save_freq=args.save_freq,
        paper_save_freq=args.paper_save_freq,
    ).train()


if __name__ == "__main__":
    main()