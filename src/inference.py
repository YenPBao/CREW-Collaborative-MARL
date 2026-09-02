# src/inference.py
"""
Inference for MARS-RWG

Generate Related Work using trained Actor networks.

Usage:
    # Default (WSN task, latest checkpoint)
    python src/inference.py
    
    # Use specific task
    python src/inference.py --task rl
    python src/inference.py --task nlp
    
    # Custom topic
    python src/inference.py --title "My Paper" --abstract "My abstract..."
    
    # Enable cold start
    python src/inference.py --cold-start
    
    # Use specific checkpoint
    python src/inference.py --ckpt ./checkpoints/self_supervised/best
"""

import os
import sys
from typing import List, Dict, Optional, Set
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict
from datetime import datetime
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rl_env.rwg_env import RWG_Enviroment
from src.networks.actor import Actor
from src.core.tasks import TASKS, get_task, list_tasks
from src.agent.model import WorkerLLM
from src.core.utils import load_papers, load_graph_in, load_graph_out
from src.core.tracker import ActionTracker


# Observation dimension: 4 documents x 5 features (see src/rl_env/rwg_env.py)
OBS_DIM = 20

# Cold start sequence
COLD_START_SEQ = [0, 0, 1, 2, 3]  # SEARCH, SEARCH, UPDATE, WRITE, FEEDBACK
ACTION_NAMES = {0: "SEARCH", 1: "UPDATE", 2: "WRITE", 3: "FEEDBACK"}

DEFAULT_CKPT_DIR = "./checkpoints/self_supervised"





def get_latest_checkpoint(ckpt_dir: str) -> str:
    """
    Get latest checkpoint (by timestamp).
    Priority: selected/ > latest timestamp
    """
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


def get_best_checkpoint(ckpt_dir: str) -> str:
    """
    Get checkpoint with highest best_sim in stats.pth.
    """
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
    best_sim = -1
    
    for c in ckpts:
        stats_f = c / "stats.pth"
        if stats_f.exists():
            try:
                stats = torch.load(stats_f, map_location="cpu", weights_only=False)
                sim = stats.get("best_sim", 0.0)
                if sim > best_sim:
                    best_sim = sim
                    best_ckpt = c
            except:
                pass
    
    if best_ckpt:
        print(f"   Using best checkpoint (sim={best_sim:.3f})")
        return str(best_ckpt)
    
    ckpts = sorted(ckpts, key=lambda x: x.name, reverse=True)
    return str(ckpts[0]) if ckpts else None


class Generator:
    """Generate Related Work using trained models."""
    
    def __init__(
        self,
        agents: int = 2,
        steps: int = 20,
        cold_start: bool = False,
        from_scratch: bool = False,
        ckpt_path: str = None,
        use_best: bool = False,
        output_dir: str = "./outputs",
        model_name: str = None,
        heuristic: bool = False,
        fixed_actions: list = None,
        preloaded_papers: Optional[Dict] = None
    ):
        self.agents_n = agents
        self.steps = steps
        self.cold_start = cold_start
        self.from_scratch = from_scratch
        self.model_name = model_name
        self.heuristic = heuristic
        self.fixed_actions = fixed_actions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Init LLM and env
        # Init LLM and env
        print("Initializing...")
        llm = WorkerLLM(model_name=model_name) if model_name else WorkerLLM()
        
        # Skip embeddings if using heuristic or random policy (not needed for actions)
        should_skip_embed = (heuristic or from_scratch)
        if should_skip_embed:
            print("   Skipping embedding model load (Heuristic/Random mode)")
            
        self.env = RWG_Enviroment(
            num_agents=agents, 
            llm=llm, 
            use_gpu=torch.cuda.is_available(),
            skip_embed=should_skip_embed
        )
        self.llm = llm  # Keep reference for token counting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and inject paper and graph data
        print("   Loading paper and citation graph data...")
        if preloaded_papers is not None:
            self.env.papers = preloaded_papers
        else:
            self.env.papers = load_papers()
        self.env.citation_adj_in = load_graph_in()
        self.env.citation_adj_out = load_graph_out()
        
        # Init actors
        self._init_actors()
        
        # Load checkpoint unless from_scratch
        self.loaded_ckpt = None
        if from_scratch:
            pass  # No checkpoint, use random actions
        else:
            if ckpt_path:
                self._load_checkpoint(ckpt_path)
                self.loaded_ckpt = ckpt_path
            elif use_best:
                ckpt = get_best_checkpoint(DEFAULT_CKPT_DIR)
                if ckpt:
                    self._load_checkpoint(ckpt)
                    self.loaded_ckpt = ckpt
            else:
                ckpt = get_latest_checkpoint(DEFAULT_CKPT_DIR)
                if ckpt:
                    self._load_checkpoint(ckpt)
                    self.loaded_ckpt = ckpt
    
    def _init_actors(self):
        """Initialize Actor networks."""
        obs_dim = 20  # Must match training (train_self_supervised.py) : Matrix 4x5
        act_dim = 4
        
        self.actors = {}
        for aid in self.env.possible_agents:
            self.actors[aid] = Actor(obs_dim, act_dim).to(self.device)
            self.actors[aid].eval()
    
    def _load_checkpoint(self, path: str):
        """Load trained actors with smarter path detection."""
        p = Path(path)
        if not p.exists():
            print(f"   Checkpoint not found: {path}")
            return
        
        # Auto-detect subdirectories if .pth files aren't in the root of path
        if not (p / "actor.pth").exists() and not (p / f"actor_{self.env.possible_agents[0]}.pth").exists():
            for sub in ["latest", "selected"]:
                if (p / sub / "actor.pth").exists() or (p / sub / f"actor_{self.env.possible_agents[0]}.pth").exists():
                    p = p / sub
                    print(f"   [INFO] Auto-detected checkpoint subdirectory: {sub}")
                    break

        print(f"   Loading: {p}")
        
        # Try shared actor first (parameter sharing)
        actor_shared = p / "actor.pth"
        if actor_shared.exists():
            print(f"   Loading shared actor.pth for all agents")
            state_dict = torch.load(actor_shared, map_location=self.device, weights_only=False)
            for aid in self.env.possible_agents:
                self.actors[aid].load_state_dict(state_dict)
                self.actors[aid].eval()
        else:
            # Try per-agent actors
            for aid in self.env.possible_agents:
                f = p / f"actor_{aid}.pth"
                if f.exists():
                    self.actors[aid].load_state_dict(torch.load(f, map_location=self.device, weights_only=False))
                    self.actors[aid].eval()
        
        stats_f = p / "stats.pth"
        if stats_f.exists():
            stats = torch.load(stats_f, map_location=self.device, weights_only=False)
            print(f"   Epoch {stats.get('epoch', '?')}, best_sim={stats.get('best_sim', 0):.3f}")
    
    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten the (4x5) observation matrix to a 20-D vector.

        Must match RWG_Environment's observation layout and the Actor's input
        dimension (see obs_dim in src/train_self_supervised.py).
        """
        if isinstance(obs, np.ndarray):
            arr = obs.flatten().astype(np.float32)
        elif isinstance(obs, dict):
            # Fallback for legacy dict format
            x = np.array(obs.get("x_features", np.zeros(15)), dtype=np.float32).flatten()[:15]
            s = np.array(obs.get("social", np.zeros(5)), dtype=np.float32).flatten()[:5]
            q = np.array(obs.get("quality", np.zeros(3)), dtype=np.float32).flatten()[:3]
            arr = np.concatenate([x, s, q]).astype(np.float32)
        else:
            arr = np.zeros(OBS_DIM, dtype=np.float32)
        if arr.size < OBS_DIM:
            arr = np.pad(arr, (0, OBS_DIM - arr.size))
        return arr[:OBS_DIM].astype(np.float32)
    
    def _select_action(self, obs: dict, aid: str) -> int:
        """Select action. Random sampling if from_scratch, otherwise deterministic."""
        if self.from_scratch:
            # Truly random action for random policy
            return np.random.randint(0, 4)
        
        # Stochastic action from trained policy
        obs_t = torch.tensor(self._flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actors[aid](obs_t)
            # print(f"    [Policy] {aid} Logits: {logits.cpu().numpy().round(3)}")
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def _heuristic_policy(self, aid: str) -> int:
        """
        Heuristic logic based on availability of knowledge:
        1. No local knowledge -> SEARCH (0)
        2. No global knowledge -> UPDATE (1)
        3. No draft -> WRITE (2)
        4. Has draft -> FEEDBACK (3) (with some randomness)
        """
        # Check if agent has local knowledge
        local_cache = self.env.local_cache.get(aid, {})
        local_memory = local_cache.get("memory", "") if isinstance(local_cache, dict) else ""
        if isinstance(local_memory, list):
            local_memory = " ".join(str(x) for x in local_memory)
        has_local = bool(str(local_memory).strip())
        
        # Check global knowledge and draft
        has_global = bool((self.env.global_knowledge or "").strip())
        has_draft = bool((self.env.draft or "").strip())

        # Weights [SEARCH, UPDATE, WRITE, FEEDBACK]
        if not has_local:
            # Agent needs to gather info first
            w = np.array([1.0, 0.0, 0.0, 0.0])
        elif not has_global:
            # Has local but needs to sync global
            w = np.array([0.30, 0.50, 0.15, 0.05])
        elif not has_draft:
            # Has global but needs draft
            w = np.array([0.15, 0.15, 0.60, 0.10])
        else:
            # Improve draft
            w = np.array([0.15, 0.15, 0.30, 0.40])
            
        # Normalize
        w = w / w.sum()
        return int(np.random.choice([0, 1, 2, 3], p=w))
    
    def generate(self, title: str, abstract: str, save: bool = True, paper_id: int = None, token_limit: int = 0, prompt_overhead: int = 800) -> Dict:
        """Generate Related Work with progress bar."""
        # Show configuration
        print("\n" + "="*60)
        print("INFERENCE CONFIGURATION")
        print("="*60)
        print(f"   Steps:      {self.steps}")
        print(f"   Agents:     {self.agents_n}")
        print(f"   Cold start: {self.cold_start}")
        print(f"   Checkpoint: {self.loaded_ckpt or 'None (random policy)'}")
        if token_limit > 0:
            print(f"   Token Limit: {token_limit} (overhead={prompt_overhead})")
        print()
        print("Topic:")
        print(f"   Title: {title[:60]}{'...' if len(title)>60 else ''}")
        print(f"   Abstract: {abstract[:100]}{'...' if len(abstract)>100 else ''}")
        print("="*60 + "\n")
        
        # Map -1 to the current target paper for the initial Search action
        # This matches the logic from test_env.py so the agents can find its citations
        target_paper = {
            "title": title,
            "abstract": abstract
        }
        self.env.papers[-1] = target_paper
        
        # Find if this paper exists in our database to get its real citations
        target_pid = paper_id
        if target_pid is None:
            for pid, p in self.env.papers.items():
                if pid != -1 and p.get("title", "").lower().strip() == title.lower().strip():
                    target_pid = pid
                    break
                
        if target_pid is not None:
            print(f"   Mapping citations from real paper ID: {target_pid}")
            self.env.citation_adj_out[-1] = self.env.citation_adj_out.get(target_pid, [])
            self.env.citation_adj_in[-1] = self.env.citation_adj_in.get(target_pid, [])
            print(f"   Found {len(self.env.citation_adj_out[-1])} citations.")
        else:
            print("   Warning: Target paper not found in graph database. Starting with empty citations.")
            self.env.citation_adj_out[-1] = []
            self.env.citation_adj_in[-1] = []

        # Reset env with max_rounds matching steps
        obs, _ = self.env.reset(
            options={"prompt": f"Write related work for: {title}",
                     "paper_title": title, "abstract": abstract,
                     "max_rounds": self.steps},
            clear_memory=True
        )
        
        # Run steps with action tracking
        # Run steps with action tracking
        tracker = ActionTracker(self.env.possible_agents, log_dir=str(self.output_dir))

        # Reset token counter before this episode
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'reset_token_counts'):
            self.llm.reset_token_counts()
        
        print("Running inference steps:")
        
        FIXED_SEQ = None
        if self.fixed_actions:
            name_to_id = {"search": 0, "update": 1, "write": 2, "feedback": 3}
            FIXED_SEQ = [name_to_id[a.lower()] for a in self.fixed_actions if a.lower() in name_to_id]
            if not FIXED_SEQ:
                FIXED_SEQ = [0, 1, 3, 2]  # Fallback
        elif getattr(self, 'fixed_seq', False): # For backwards compatibility with CLI
            FIXED_SEQ = [0, 1, 3, 2]

        for step in range(self.steps):
            actions = {}
            for aid in self.env.possible_agents:
                if FIXED_SEQ:
                    actions[aid] = FIXED_SEQ[step % len(FIXED_SEQ)]
                elif self.cold_start and step < len(COLD_START_SEQ):
                    actions[aid] = COLD_START_SEQ[step]
                elif self.from_scratch:
                     # Explicitly random policy (overrides heuristic)
                    actions[aid] = self._select_action(obs[aid], aid)
                elif self.heuristic:
                    actions[aid] = self._heuristic_policy(aid)
                else:
                    actions[aid] = self._select_action(obs[aid], aid)
            
            obs, rewards, terms, _, _ = self.env.step(actions)
            
            # Log and print this step with rewards
            tracker.log_step(step, actions, env=self.env, rewards=rewards)
            tracker.save_snapshot(self.env, step)
            
            # Token Limit Check
            if token_limit > 0 and self.llm:
                stats = self.llm.get_token_counts()
                raw_in = stats.get("prompt_tokens", 0)
                n_calls = stats.get("num_calls", 0)
                out = stats.get("completion_tokens", 0)
                adj_in = max(0, raw_in - n_calls * prompt_overhead)
                current_total = adj_in + out
                if current_total >= token_limit:
                    print(f"\n   [Token Limit] Reached {current_total} >= {token_limit}. Stopping at step {step+1}.")
                    break

            if all(terms.values()):
                print(f"   (Episode terminated at step {step+1})")
                break
        
        # Show action summary
        tracker.print_summary()
        
        # Get outputs - draft only (outline is implicit in draft now)
        outline = "*No outline step*"
        # draft = getattr(self.env.state, "draft", "") or ""
        # FIX: Access env.draft directly
        draft = self.env.draft or ""
        
        result = {
            "title": title,
            "abstract": abstract,
            "outline": outline,
            "draft": draft,
            "steps": self.steps,
            "action_counts": {}, # tracker.counts, # Removed local counts access
            "timestamp": datetime.now().isoformat()
        }

        # Attach token usage for this episode
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_token_counts'):
            result["token_usage"] = self.llm.get_token_counts()
        
        if save:
            result["file"] = str(self._save(result))
        
        print(f"\nGENERATION COMPLETE!")
        # Count topics in outline (lines starting with "Topic")
        outline_topics = len([l for l in outline.split('\n') if l.strip().startswith('Topic')]) if outline else 0
        print(f"   Outline: {outline_topics} topics")
        print(f"   Draft: {len(draft.split())} words ({len(draft)} chars)")
        if save:
            print(f"   Saved: {result['file']}")
        
        # Show first 1-2 sentences preview
        if draft:
            sentences = draft.split('.')
            preview = '. '.join(sentences[:2]).strip()
            if preview and not preview.endswith('.'):
                preview += '.'
            print(f"\nDraft Preview: \"{preview}\"")
        
        return result
    
    def _save(self, result: Dict) -> Path:
        """Save to markdown."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"related_work_{ts}.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Generated Related Work\n\n")
            f.write(f"**Title:** {result['title']}\n\n")
            f.write(f"**Abstract:** {result['abstract']}\n\n")
            f.write(f"## Outline\n\n")
            if result['outline']:
                f.write(f"{result['outline']}\n")
            else:
                f.write("*No outline*\n")
            f.write(f"\n## Draft\n\n{result['draft'] or '*No draft*'}\n")
            f.write(f"\n---\nGenerated: {result['timestamp']}\n")
            f.write(f"\n---\nGenerated: {result['timestamp']}\n")
        
        return path
    
    def get_llm_token_counts(self) -> dict:
        """Return accumulated token counts from the LLM for the last episode."""
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_token_counts'):
            return self.llm.get_token_counts()
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate_from_task(self, task_name: str = "wsn") -> Dict:
        """Generate from predefined task."""
        task = get_task(task_name)
        print(f"\nTask: {task['name']}")
        return self.generate(task["paper_title"], task["abstract"])


def main():
    parser = argparse.ArgumentParser(description="Generate Related Work (MARS-RWG)")
    
    # Hardcoded defaults
    default_title = "Multi-agent reinforcement learning based general charging model for maximizing target coverage and connectivity"
    default_abstract = """Wireless rechargeable sensor networks (WRSNs) employ mobile chargers (MCs) to replenish sensor energy, yet designing charging strategies that prolong network lifetime while preserving coverage and connectivity remains challenging. Offline (periodic) tours are inflexible to heterogeneous, time-varying consumption, whereas online (on-demand) methods must simultaneously decide the next charging location and the delivered energy, and they often assume identical sensor roles or rely on full-charge policies, leading to long waits and premature MC returns. We address these limitations with a generalized multi–mobile-charger framework that supports multi-point charging, allowing an MC to replenish multiple sensors per stop. We formulate cooperative control as a Decentralized Partially Observable Semi-Markov Decision Process (Dec-POSMDP) to handle partial observations and variable action durations. A compact observation encoding—distribution descriptors with a spatial heatmap—enables efficient reasoning and transfer across network instances. We then learn decentralized policies via an asynchronous multi-agent reinforcement learning solver based on Independent Proximal Policy Optimization (IPPO). Experiments on large-scale WRSNs show that the proposed method consistently extends network lifetime while maintaining target coverage and connectivity, and reduces charging delay and MC energy waste compared with state-of-the-art baselines."""
    
    parser.add_argument("--task", "-t", type=str, default=None,
                        help=f"Built-in task: {list(TASKS.keys())}. Ignored when --title and --abstract are given.")
    parser.add_argument("--title", type=str, default=None, help="Custom title (requires --abstract)")
    parser.add_argument("--abstract", type=str, default=None, help="Custom abstract (requires --title)")
    parser.add_argument("-a", type=int, default=4, help="Agents (default: 4)")
    parser.add_argument("-s", type=int, default=15, help="Steps (default: 15)")
    parser.add_argument("--cold-start", action="store_true", help="Enable cold start")
    parser.add_argument("--scratch", action="store_true", help="Use random policy (no checkpoint)")
    parser.add_argument("--best", action="store_true", help="Use best checkpoint (highest sim)")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint path")
    parser.add_argument("--output", type=str, default="./outputs", help="Output dir")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--model", type=str, default=None, help="LLM model (default: gpt-4.1-mini, or 'qwen' for local)")
    parser.add_argument("--heuristic", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the heuristic policy instead of the trained actor. "
                             "Pass --no-heuristic to select actions with the loaded checkpoint (default: --heuristic)")
    parser.add_argument("--fixed-seq", action="store_true", help="Force sequence: SEARCH, UPDATE, FEEDBACK, WRITE")
    
    args = parser.parse_args()
    
    if args.list_tasks:
        list_tasks()
        return
    
    gen = Generator(
        agents=args.a,
        steps=args.s,
        cold_start=args.cold_start,
        from_scratch=args.scratch,
        ckpt_path=args.ckpt,
        use_best=args.best,
        output_dir=args.output,
        model_name=args.model,
        heuristic=args.heuristic,
        fixed_actions=["search", "update", "feedback", "write"] if args.fixed_seq else None
    )
    
    if args.title and args.abstract:
        result = gen.generate(args.title, args.abstract)
    elif args.task:
        result = gen.generate_from_task(args.task)
    else:
        result = gen.generate(default_title, default_abstract)


if __name__ == "__main__":
    main()
