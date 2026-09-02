import os
import shutil
import json
from typing import Dict, Any, List

class ActionTracker:
    def __init__(self, possible_agents: List[str], log_dir: str = "logs"):
        self.possible_agents = possible_agents
        self.log_dir = log_dir
        self.action_names = {0: "SEARCH", 1: "UPDATE", 2: "WRITE", 3: "FEEDBACK"}
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def to_text(self, x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)

    def pretty_preview(self, text: str, head: int = 600, tail: int = 300) -> str:
        text = text or ""
        if len(text) <= head + tail + 50:
            return text
        return text[:head] + "\n...\n" + text[-tail:]

    def log_step(self, step: int, actions: Dict[str, int], env=None, rewards: Dict[str, float] = None):
        print(f"\n=== STEP {step} ===")
        
        # Print actions in a table-like format
        print(f"{'Agent':<10} | {'Action':<10} | {'Reward':<10}")
        print("-" * 36)
        
        for aid in self.possible_agents:
            act_idx = actions.get(aid, -1)
            act_name = self.action_names.get(act_idx, "UNKNOWN")
            
            rew_str = "-"
            if rewards and aid in rewards:
                rew_str = f"{rewards[aid]:.4f}"
                
            print(f"{aid:<10} | {act_name:<10} | {rew_str:<10}")

        if env:
            # Print stats
            draft_len = len((env.draft or "").split())
            global_len = len((env.global_knowledge or "").split())
            local_counts = []
            for aid in self.possible_agents:
                mem = env.local_cache.get(aid, {}).get("memory", "")
                l_len = len(mem.split()) if mem else 0
                local_counts.append(f"{aid}: {l_len}")
            local_str = ", ".join(local_counts)
            print(f"\nStats: Draft Words={draft_len} | Global Knowledge Words={global_len} | Local Memory Words ({local_str})")
            
            # Preview Draft if exists
            if (env.draft or "").strip():
                print("\n--- DRAFT PREVIEW ---")
                print(self.pretty_preview(env.draft))
                print("-" * 30)

    def save_snapshot(self, env, step_idx: int):
        """Save draft, global, local memories to separate files."""
        # Clean specific step files if needed, currently just overwrites
        
        # Save Draft and Global
        with open(os.path.join(self.log_dir, f"draft_step{step_idx}.txt"), "w", encoding="utf-8") as f:
            f.write(self.to_text(env.draft))

        with open(os.path.join(self.log_dir, f"global_step{step_idx}.txt"), "w", encoding="utf-8") as f:
            f.write(self.to_text(env.global_knowledge))

        with open(os.path.join(self.log_dir, f"feedback_step{step_idx}.txt"), "w", encoding="utf-8") as f:
            f.write(self.to_text(env.feedback))

        # Save Local Memories
        for aid in self.possible_agents:
            mem = self.to_text(env.local_cache.get(aid, {}).get("memory", ""))
            with open(os.path.join(self.log_dir, f"local_{aid}_step{step_idx}.txt"), "w", encoding="utf-8") as f:
                f.write(mem)

    def print_summary(self):
        print("\n" + "="*40)
        print(f"Logs saved to: {os.path.abspath(self.log_dir)}")
        print("="*40 + "\n")
