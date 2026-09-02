#!/usr/bin/env python3
# src/train_ablation_study.py
"""
Ablation Study Training for MARS-RWG
=====================================

Three ablations vs the full MARS-RWG system:
  A) no_feedback    — Remove FEEDBACK action. 3-action env, obs (3x5)=15 dims.
  B) no_comp_reward — Non-WRITE reward = delta_Q_D only (no component quality signal).
  C) no_social      — Zero out social column (col 4) in (4x5)=20-dim observation.

Checkpoints saved every 6 papers in separate directories:
  checkpoints/ablation_no_feedback/
  checkpoints/ablation_no_comp_reward/
  checkpoints/ablation_no_social/

Training logs written to:
  run/ablation_no_feedback.log
  run/ablation_no_comp_reward.log
  run/ablation_no_social.log

================================================================================
GPU Training Commands (Windows PowerShell)
================================================================================

# 1. Activate conda environment first:
#    conda activate rwg

# 2. Run one ablation at a time with live log streaming:

# Ablation A - No Feedback:
#   $env:PYTHONUNBUFFERED=1; python src/train_ablation_study.py --ablation no_feedback --limit 30 --cold-start --scratch

# Ablation B - No Component Reward:
#   $env:PYTHONUNBUFFERED=1; python src/train_ablation_study.py --ablation no_comp_reward --limit 30 --cold-start --scratch

# Ablation C - No Social Observation:
#   $env:PYTHONUNBUFFERED=1; python src/train_ablation_study.py --ablation no_social --limit 30 --cold-start --scratch

# 3. Run ALL three sequentially (logs go to separate files automatically):
#   $env:PYTHONUNBUFFERED=1; python src/train_ablation_study.py --ablation all --limit 30 --cold-start --scratch

# 4. Redirect stdout to additional log file if desired:
#   $env:PYTHONUNBUFFERED=1; python src/train_ablation_study.py --ablation all --limit 30 --cold-start --scratch 2>&1 | Tee-Object -FilePath run/ablation_all_stdout.log

# Linux / Mac:
#   PYTHONUNBUFFERED=1 python src/train_ablation_study.py --ablation all --limit 30 --cold-start --scratch 2>&1 | tee run/ablation_all.log

# 5. Useful flags:
#   --ablation   : no_feedback | no_comp_reward | no_social | all  (default: all)
#   -a           : number of agents (default: 2)
#   -e           : number of epochs (default: 5)
#   -s           : steps per paper (default: 15)
#   --limit      : max papers to train on (default: 30)
#   --cold-start : use heuristic warm-up steps
#   --scratch    : train from scratch (no checkpoint loading)
#   --model      : LLM model name override
#   --paper-save-freq : checkpoint every N papers (default: 6)
================================================================================
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from torch.distributions import Categorical
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_self_supervised import (
    Trainer,
    build_cited_paper,
    compute_similarity,
    ACTION_NAMES,
)
from src.rl_env.rwg_env_no_feedback import RWG_Environment_NoFeedback
from src.networks.actor import Actor
from src.networks.critic import Critic
from src.core.tracker import ActionTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_SAVE_FREQ = 6          # Save checkpoint every N papers
HEURISTIC_STEPS_NO_FB = 3   # No-feedback cold start: SEARCH -> UPDATE -> WRITE
ACTION_NAMES_NO_FB = {0: "SEARCH", 1: "UPDATE", 2: "WRITE"}


# ---------------------------------------------------------------------------
# File logger
# ---------------------------------------------------------------------------
def setup_file_logger(log_path: str) -> logging.Logger:
    """Logger that writes to file AND stdout simultaneously."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    name = f"ablation.{os.path.basename(log_path)}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# AblationTrainer — base for all three ablations
# ---------------------------------------------------------------------------
class AblationTrainer(Trainer):
    """
    Extends Trainer with:
      - Per-paper checkpoint saves every paper_save_freq papers.
      - Separate log file per ablation.
      - Fixed _train_paper: passes self.papers_db / self.graph_out to
        build_cited_paper (original code had undefined variable 'papers').
      - Hook methods _heuristic_steps() and _action_names() so subclasses
        can change behavior without duplicating the full _train_paper loop.
    """

    ABLATION_NAME: str = "base"
    LOG_PATH: str = "run/ablation_base.log"
    CKPT_DIR: str = "checkpoints/ablation_base"

    def __init__(self, paper_save_freq: int = PAPER_SAVE_FREQ, **kwargs):
        kwargs.setdefault("ckpt_dir", self.CKPT_DIR)
        super().__init__(**kwargs)
        self.paper_save_freq = paper_save_freq
        self.total_papers_processed = 0
        os.makedirs("run", exist_ok=True)
        self.log = setup_file_logger(self.LOG_PATH)
        self.log.info(
            f"=== [{self.ABLATION_NAME}] ready | "
            f"{len(self.paper_ids)} papers | {self.epochs} epochs ==="
        )

    # ------------------------------------------------------------------
    # Hook methods — override in subclasses as needed
    # ------------------------------------------------------------------
    def _heuristic_steps(self) -> int:
        """Number of forced heuristic warm-up steps per paper."""
        return 4  # default: SEARCH -> UPDATE -> FEEDBACK -> WRITE

    def _action_names(self) -> Dict[int, str]:
        """Map from action int to name string for logging."""
        return ACTION_NAMES

    # ------------------------------------------------------------------
    # Override train() to add per-paper checkpointing + structured logging
    # ------------------------------------------------------------------
    def train(self):
        self.log.info(
            f"Training started | papers={len(self.paper_ids)} "
            f"epochs={self.epochs} steps={self.steps} agents={self.agents}"
        )
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        all_scores_flat: List[float] = []
        epoch_scores_list: List[List[float]] = []

        epoch_pbar = tqdm(
            range(self.epochs),
            desc=f"  [{self.ABLATION_NAME}] Epochs",
            position=0,
            leave=True,
        )

        for epoch in epoch_pbar:
            np.random.shuffle(self.paper_ids)
            sims: List[float] = []
            epoch_rewards: List[float] = []
            epoch_actor_losses: List[float] = []
            epoch_critic_losses: List[float] = []
            epoch_entropies: List[float] = []

            paper_pbar = tqdm(
                self.paper_ids,
                desc=f"   Epoch {epoch + 1}",
                position=1,
                leave=False,
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

                # Per-paper checkpoint every paper_save_freq papers
                if self.total_papers_processed % self.paper_save_freq == 0:
                    self._save_checkpoint_paper(self.total_papers_processed)

                paper_pbar.set_postfix_str(
                    f"sim={sim:.3f} | avg={np.mean(sims):.3f} | "
                    f"r={paper_info['avg_step_reward']:.3f} | "
                    f"p={self.total_papers_processed}"
                )

                self.log.info(
                    f"[E{epoch + 1}] p={self.total_papers_processed} pid={pid} "
                    f"sim={sim:.4f} r={paper_info['avg_step_reward']:.4f} "
                    f"aloss={loss_info['actor_loss']:.4f} closs={loss_info['critic_loss']:.4f} "
                    f"words={paper_info['draft_words']} neighbors={paper_info['neighbors']}"
                )

            paper_pbar.close()
            epoch_scores_list.append(sims)

            avg_sim = float(np.mean(sims)) if sims else 0.0
            avg_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            avg_al = float(np.mean(epoch_actor_losses)) if epoch_actor_losses else 0.0
            avg_cl = float(np.mean(epoch_critic_losses)) if epoch_critic_losses else 0.0
            avg_ent = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0

            if avg_sim > self.best_sim:
                self.best_sim = avg_sim

            # Always save epoch checkpoint
            self._save_checkpoint(epoch=epoch)

            msg = (
                f"[Epoch {epoch + 1}] avg_sim={avg_sim:.4f} avg_r={avg_r:.4f} "
                f"actor_loss={avg_al:.4f} critic_loss={avg_cl:.4f} entropy={avg_ent:.4f} "
                f"total_papers={self.total_papers_processed} best_sim={self.best_sim:.4f}"
            )
            self.log.info(msg)
            tqdm.write(f"   [OK] [{self.ABLATION_NAME}] " + msg)

        epoch_pbar.close()
        self.log.info(f"Training complete. Best sim: {self.best_sim:.4f}")
        print(f"\n  [{self.ABLATION_NAME}] TRAINING COMPLETE! Best sim: {self.best_sim:.4f}\n")

    # ------------------------------------------------------------------
    # Bug-fixed _train_paper (uses self.papers_db / self.graph_out,
    # hooks _heuristic_steps and _action_names for subclass flexibility)
    # ------------------------------------------------------------------
    def _train_paper(self, pid: int) -> Dict[str, Any]:
        paper = self.papers_db[pid]
        gold = (paper.get("related_work") or "").strip()
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        real_citations = self.graph_out.get(pid, []) or []
        valid_neighbors = [x for x in real_citations if x in self.papers_db]

        # Reuse shared dict; add -1 sentinel for the target paper
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
            }
            # NOTE: clear_memory not passed — not available in no-feedback env;
            # env.reset() clears all state unconditionally anyway.
        )
        self.env.list_actions = []  # env.reset() does not clear this; action counts accumulate otherwise

        for aid in self.env.possible_agents:
            self.env.read_history[aid] = [{"id": -1}]

        # FIX: original code used undefined 'papers'/'graph_out' locals;
        # use self.papers_db and self.graph_out instead.
        try:
            self.env.cited_paper = build_cited_paper(
                pid, self.papers_db, self.graph_out, max_refs=10
            )
        except Exception:
            pass

        step_rewards: List[float] = []
        step_sims: List[float] = []
        heuristic_steps = self._heuristic_steps()
        action_names = self._action_names()

        for step in range(self.steps):
            actions: Dict[str, int] = {}
            current_actions: Dict[str, int] = {}
            lps: Dict[str, float] = {}
            vals: Dict[str, float] = {}

            for aid in self.env.possible_agents:
                if self.cold_start and step < heuristic_steps:
                    act = self._get_heuristic_action(aid)
                    obs_t = torch.tensor(
                        self._flatten_obs(obs[aid]),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.shared_actor(obs_t)
                        dist = Categorical(logits=logits)
                        lps[aid] = float(
                            dist.log_prob(torch.tensor([act], device=self.device)).item()
                        )
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

            rewards_str = ", ".join(f"{a}: {r:.4f}" for a, r in rewards.items())
            act_str = {a: action_names.get(v, str(v)) for a, v in current_actions.items()}

            print("\n" + "-" * 60)
            print(f" [STEP] {step + 1}/{self.steps} | Paper: {pid} | global_step: {self.global_env_step}")
            print(f" Title   : {title[:70]}...")
            print(f" Actions : {act_str}")
            print(f" Rewards : {rewards_str} (mean: {reward_mean:.4f})")
            print(f" Sim     : {sim:.4f} | Words: {len(draft.split())}")
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

        # Bootstrap last value
        for aid in self.env.possible_agents:
            if aid in obs:
                obs_t = torch.tensor(
                    self._flatten_obs(obs[aid]), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    self.buffer.data[aid]["values"].append(
                        float(self.shared_critic(obs_t).item())
                    )
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

    # ------------------------------------------------------------------
    # Per-paper checkpoint
    # ------------------------------------------------------------------
    def _save_checkpoint_paper(self, paper_count: int):
        """Save actor/critic weights keyed by total papers processed."""
        base_path = self.ckpt_dir / self.run_timestamp
        p = base_path / f"paper_{paper_count}"
        p.mkdir(parents=True, exist_ok=True)

        torch.save(self.shared_actor.state_dict(), p / "actor.pth")
        torch.save(self.shared_critic.state_dict(), p / "critic.pth")
        torch.save(
            {
                "ablation": self.ABLATION_NAME,
                "timestamp": self.run_timestamp,
                "best_sim": self.best_sim,
                "total_papers": paper_count,
                "global_env_step": self.global_env_step,
                "global_agent_step": self.global_agent_step,
            },
            p / "stats.pth",
        )
        self.log.info(f"[CKPT] paper_{paper_count} → {p}")
        print(f"   [SAVE] Paper checkpoint: {p}")


# ===========================================================================
# Ablation A: No FEEDBACK
# ===========================================================================
class TrainerNoFeedback(AblationTrainer):
    """
    Ablation A: Remove the FEEDBACK action.

    Uses RWG_Environment_NoFeedback (3 actions: SEARCH, UPDATE, WRITE).
    Observation matrix is (3 x 5) = 15 dims (no FEEDBACK row).
    Cold start: SEARCH -> UPDATE -> WRITE  (3 heuristic steps).
    """

    ABLATION_NAME = "no_feedback"
    LOG_PATH = "run/ablation_no_feedback.log"
    CKPT_DIR = "checkpoints/ablation_no_feedback"

    def __init__(self, **kwargs):
        # Incompatible obs/act dims with full model — always start from scratch
        kwargs["from_scratch"] = True
        super().__init__(**kwargs)

        # Replace env created by base Trainer with the 3-action no-feedback variant.
        # Reuse the LLM already instantiated inside the old env (avoid double init).
        old_llm = self.env.llm
        self.env = RWG_Environment_NoFeedback(
            num_agents=self.agents,
            llm=old_llm,
            use_gpu=torch.cuda.is_available(),
            papers=self.papers_db,
            citation_adj_in=self.graph_in,
            citation_adj_out=self.graph_out,
        )

        # Reinit tracker to point at the new env's agent list
        self.tracker = ActionTracker(
            self.env.possible_agents, log_dir=self.log_dir
        )

        self.log.info("No-feedback env active (3 actions, obs_dim=15)")

    # _init_networks is called by super().__init__() via Python MRO.
    # By overriding it here, the correct 15-dim / 3-action networks are created
    # during base class init — no second call needed.
    def _init_networks(self):
        obs_dim = 15  # 3 rows x 5 cols
        act_dim = 3
        self.shared_actor = Actor(obs_dim, act_dim).to(self.device)
        self.shared_critic = Critic(obs_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.shared_actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.shared_critic.parameters(), lr=1e-3)
        print(f"   [OK] NoFeedback Networks: obs={obs_dim}, act={act_dim}, hidden=32")

    def _heuristic_steps(self) -> int:
        return HEURISTIC_STEPS_NO_FB  # 3: SEARCH -> UPDATE -> WRITE

    def _action_names(self) -> Dict[int, str]:
        return ACTION_NAMES_NO_FB

    def _get_heuristic_action(self, aid: str) -> int:
        """SEARCH -> UPDATE -> WRITE (no FEEDBACK step)."""
        mem = (self.env.local_cache.get(aid) or {}).get("memory", "")
        if not str(mem).strip():
            return 0  # SEARCH
        if not (self.env.global_knowledge or "").strip():
            return 1  # UPDATE
        return 2  # WRITE

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Flatten (3, 5) obs matrix to 15-dim vector."""
        if isinstance(obs, np.ndarray):
            arr = obs.flatten().astype(np.float32)
            if arr.size < 15:
                arr = np.pad(arr, (0, 15 - arr.size))
            return arr[:15]
        return np.zeros(15, dtype=np.float32)

    def _select_action(self, obs: Any, aid: str) -> Tuple[int, float, float]:
        """Sample from 3-action network."""
        obs_t = torch.tensor(
            self._flatten_obs(obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.shared_actor(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            a = int(action.item())
            if a >= 3:
                a = 0
            logp = float(dist.log_prob(torch.tensor(a, device=self.device)).item())
            val = float(self.shared_critic(obs_t).item())
        return a, logp, val


# ===========================================================================
# Ablation B: No Component Reward
# ===========================================================================
class TrainerNoCompReward(AblationTrainer):
    """
    Ablation B: Remove the component (per-action) quality signal from reward.

    All agents receive only draft-quality signal:
      r = normalize(delta_Q_D)  for WRITE
      r = normalize(delta_Q_D)  for non-WRITE  (instead of 0.1*dQD + 0.9*dQk)

    Implemented by setting env.draft_ambient_gamma = 1.0:
      r_total = gamma * draft_gain + (1-gamma) * comp_gain
      With gamma=1.0 -> r_total = draft_gain  (comp_gain zeroed out)
    """

    ABLATION_NAME = "no_comp_reward"
    LOG_PATH = "run/ablation_no_comp_reward.log"
    CKPT_DIR = "checkpoints/ablation_no_comp_reward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # draft_ambient_gamma=1.0 makes r_total = 1.0*draft_gain + 0.0*comp_gain
        self.env.draft_ambient_gamma = 1.0
        self.log.info(
            "No-comp-reward mode: draft_ambient_gamma set to 1.0 "
            "(non-WRITE reward = delta_Q_D only)"
        )


# ===========================================================================
# Ablation C: No Social Observation
# ===========================================================================
class TrainerNoSocial(AblationTrainer):
    """
    Ablation C: Remove the social (activity) column from the observation.

    The (4 x 5) obs matrix has columns: [Rel, Nov/Cov, Impact, TokenBudget, Social].
    This ablation zeros out column 4 (Social) before feeding obs to the network.
    Flat indices in the 20-dim vector: 4, 9, 14, 19.

    The network still has obs_dim=20 — it just always sees social=0.
    """

    ABLATION_NAME = "no_social"
    LOG_PATH = "run/ablation_no_social.log"
    CKPT_DIR = "checkpoints/ablation_no_social"

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Flatten (4, 5) to 20-dim, then zero out the social column (col 4)."""
        arr = super()._flatten_obs(obs)  # 20-dim from base Trainer
        # Social column is index 4 in each 5-wide row: flat positions 4, 9, 14, 19
        arr[4] = 0.0
        arr[9] = 0.0
        arr[14] = 0.0
        arr[19] = 0.0
        return arr


# ===========================================================================
# Entry point
# ===========================================================================
ABLATION_MAP = {
    "no_feedback": TrainerNoFeedback,
    "no_comp_reward": TrainerNoCompReward,
    "no_social": TrainerNoSocial,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run MARS-RWG ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=list(ABLATION_MAP.keys()) + ["all"],
        help="Which ablation to run (default: all)",
    )
    parser.add_argument("-a", type=int, default=3, help="Number of agents (default: 3)")
    parser.add_argument("-e", type=int, default=3, help="Epochs (default: 3)")
    parser.add_argument("-s", type=int, default=15, help="Steps per paper (default: 15)")
    parser.add_argument("--limit", type=int, default=30, help="Max training papers (default: 30)")
    parser.add_argument("--cold-start", action="store_true", help="Use heuristic warm-up steps")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (ignore checkpoints)")
    parser.add_argument("--model", type=str, default=None, help="LLM model name override")
    parser.add_argument(
        "--paper-save-freq",
        type=int,
        default=PAPER_SAVE_FREQ,
        help=f"Checkpoint every N papers (default: {PAPER_SAVE_FREQ})",
    )
    parser.add_argument("--min-neighbors", type=int, default=5, help="Min citation neighbors (default: 5)")
    args = parser.parse_args()

    # Shared kwargs passed to every ablation trainer
    common = dict(
        agents=args.a,
        epochs=args.e,
        steps=args.s,
        limit=args.limit,
        cold_start=args.cold_start,
        from_scratch=args.scratch,
        model_name=args.model,
        paper_save_freq=args.paper_save_freq,
        min_neighbors=args.min_neighbors,
        use_wandb=False,   # Ablation runs don't use wandb by default
        save_freq=1,       # Also save epoch-level checkpoints every epoch
    )

    ablations_to_run = list(ABLATION_MAP.keys()) if args.ablation == "all" else [args.ablation]

    for name in ablations_to_run:
        cls = ABLATION_MAP[name]
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  ABLATION: {name}")
        print(f"  Class:    {cls.__name__}")
        print(f"  Ckpt dir: {cls.CKPT_DIR}")
        print(f"  Log:      {cls.LOG_PATH}")
        print(f"{sep}\n")

        trainer = cls(**common)
        trainer.train()

        print(f"\n  [{name}] Done.\n")


if __name__ == "__main__":
    main()
