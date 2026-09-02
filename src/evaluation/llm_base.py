import json
import os,re
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import numpy as np
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.utils import to_json,load_papers
from src.agent.model import WorkerLLM
from src.agent.prompts import EVALUATION_SYSTEM, EVALUATION_USER


class RelatedWorkEvaluator:
    def __init__(self, model_name: Optional[str] = None, llm: Optional[WorkerLLM] = None):
        """Initialize evaluator with specific model or LLM instance.
        
        Args:
            model_name: Model to use (e.g. 'gpt-4.1-mini', 'gemini-pro')
            llm: Pre-initialized WorkerLLM instance (overrides model_name)
        """
        if llm:
            self.llm = llm
        elif model_name:
            self.llm = WorkerLLM(model_name=model_name)
        else:
            self.llm = WorkerLLM()
        
        self.model_name = model_name or getattr(self.llm, 'model_name', 'default')
        self.metrics = ["Coverage", "Logic", "Relevance", "Citation"]
        
    def evaluate_related_work(
        self,
        related_work: str,
        abstract: str,
        cited_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        results = {
            "related_work": related_work,
            "abstract": abstract,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "total_tokens": 0,
            "model_name": self.model_name
        }
        
        user_prompt = EVALUATION_USER.format(
            all_papers=json.dumps(cited_papers, ensure_ascii=False, indent=2),
            abstract=abstract,
            context=related_work
        )

        response, tokens = self.llm.generate(user_prompt, EVALUATION_SYSTEM)
        results["total_tokens"] = tokens
        
        try:
            eval_data = to_json(response)
        except:
            print(f"Error parsing evaluation JSON: {response[:200]}...")
            eval_data = {}

        if not isinstance(eval_data, dict):
            print(f"DEBUG: Non-dict eval_data from {self.model_name}: {eval_data}")
            eval_data = {"score": str(eval_data), "reason": "Evaluator returned non-dictionary results"}
        else:
            print(f"DEBUG: Parsed JSON from {self.model_name}: {list(eval_data.keys())}")

        # Normalize keys to be case-insensitive and handle flat results
        eval_data_lower = {k.lower(): v for k, v in eval_data.items()}
        
        for metric in self.metrics:
            m_key = metric.lower()
            m_data = eval_data_lower.get(m_key, {})
            
            # If the LLM returned a flat structure for some reason, try to find it
            if not isinstance(m_data, dict):
                 raw_score = m_data
                 reason = "No reason provided (Flat JSON)."
            else:
                 raw_score = m_data.get("score", 0)
                 reason = m_data.get("reason", "No reason provided.")

            # Normalize score -> int
            if isinstance(raw_score, str):
                m = re.search(r"\d+", raw_score)
                raw_score = int(m.group()) if m else 0
            elif isinstance(raw_score, (float, np.floating)):
                raw_score = int(raw_score)
            elif not isinstance(raw_score, int):
                raw_score = 0

            results["metrics"][metric] = {
                "score": raw_score,
                "reason": reason
            }
        
        scores = [results["metrics"][m]["score"] for m in self.metrics]
        results["average_score"] = np.mean(scores) if scores else 0
        return results

    def evaluate_all_metrics(
        self,
        paper: Dict[str, Any],
        draft: str,
        cited_papers: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Evaluate all 4 metrics in a single pass."""
        if not (draft or "").strip():
             # If draft is empty, return 1 for all metrics
             return {m.lower(): 1 for m in self.metrics}, 0

        if model_name and model_name != self.model_name:
            # Temporarily switch model if requested
            old_model = self.llm.model_name
            self.llm.model_name = model_name
        
        try:
            res = self.evaluate_related_work(draft, paper["abstract"], cited_papers)
            scores = {m.lower(): data["score"] for m, data in res["metrics"].items()}
            return scores, res["total_tokens"]
        finally:
            if model_name and model_name != self.model_name:
                self.llm.model_name = old_model
    
    @staticmethod
    def evaluate_with_models(
        model_names: List[str],
        related_work: str,
        abstract: str,
        cited_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate with multiple models and calculate ensemble average.
        
        Args:
            model_names: List of model names to use (e.g. ['gpt-4.1-mini', 'gemini-pro'])
            related_work: Related work text
            abstract: Paper abstract
            cited_papers: List of cited papers
            
        Returns:
            Dict with individual model results and ensemble average
        """
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL EVALUATION ({len(model_names)} models)")
        print(f"{'='*80}\n")
        
        all_results = {}
        model_scores = {metric: [] for metric in ["Coverage", "Logic", "Relevance", "Citation"]}
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n[{i}/{len(model_names)}] Evaluating with {model_name}...")
            print("-" * 80)
            
            evaluator = RelatedWorkEvaluator(model_name=model_name)
            results = evaluator.evaluate_related_work(related_work, abstract, cited_papers)
            
            all_results[model_name] = results
            
            # Collect scores for averaging
            for metric in model_scores.keys():
                model_scores[metric].append(results["metrics"][metric]["score"])
            
            print(f"\n  Results from {model_name}:")
            for metric, data in results["metrics"].items():
                print(f"    {metric:12s}: {data['score']}/5")
            print(f"    {'Average':12s}: {results['average_score']:.2f}/5")
        
        # Calculate ensemble average
        ensemble_metrics = {}
        for metric, scores in model_scores.items():
            ensemble_metrics[metric] = {
                "score": np.mean(scores),
                "std": np.std(scores),
                "scores_by_model": {model_names[i]: scores[i] for i in range(len(model_names))}
            }
        
        ensemble_avg = np.mean([ensemble_metrics[m]["score"] for m in ensemble_metrics.keys()])
        
        ensemble_result = {
            "related_work": related_work,
            "abstract": abstract,
            "timestamp": datetime.now().isoformat(),
            "model_names": model_names,
            "individual_results": all_results,
            "ensemble_metrics": ensemble_metrics,
            "ensemble_average": ensemble_avg,
            "total_tokens": sum(r["total_tokens"] for r in all_results.values())
        }
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE RESULTS (Average across {len(model_names)} models)")
        print(f"{'='*80}\n")
        for metric, data in ensemble_metrics.items():
            print(f"  {metric:12s}: {data['score']:.2f}/5 (±{data['std']:.2f})")
        print(f"\n  {'ENSEMBLE AVG':12s}: {ensemble_avg:.2f}/5")
        print(f"\n  Total tokens used: {ensemble_result['total_tokens']:,}")
        print(f"\n{'='*80}\n")
        
        return ensemble_result


def plot_metrics_bar(
    results: Dict[str, Dict[str, Any]], 
    save_path: Optional[str] = None,
    title: str = "Related Work Evaluation Metrics"
) -> None:
    """
    Create bar chart comparing scores across metrics.
    
    Args:
        results: Dict mapping experiment names to evaluation results
        save_path: Path to save plot (if None, shows interactively)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["Coverage", "Logic", "Relevance"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)  # Width of bars
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, (exp_name, exp_results) in enumerate(results.items()):
        scores = [exp_results["metrics"][m]["score"] for m in metrics]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=exp_name, color=colors[i])
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (1-5)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_history(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Metrics Evolution Over Time"
) -> None:
    """
    Create line plot showing metric evolution over episodes/iterations.
    
    Args:
        history: List of evaluation results over time
        save_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ["Coverage", "Logic", "Relevance"]
    colors = {'Coverage': '#FF6B6B', 'Logic': '#4ECDC4', 'Relevance': '#45B7D1'}
    
    for metric in metrics:
        scores = [h["metrics"][metric]["score"] for h in history]
        episodes = list(range(1, len(scores) + 1))
        ax.plot(episodes, scores, marker='o', linewidth=2, 
                label=metric, color=colors[metric], markersize=6)
    
    ax.set_xlabel('Episode / Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (1-5)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_radar_chart(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Metrics Radar Chart"
) -> None:
    """
    Create radar/spider chart for multi-dimensional metric view.
    
    Args:
        results: Dict mapping experiment names to evaluation results
        save_path: Path to save plot
        title: Plot title
    """
    metrics = ["Coverage", "Logic", "Relevance"]
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for i, (exp_name, exp_results) in enumerate(results.items()):
        scores = [exp_results["metrics"][m]["score"] for m in metrics]
        scores += scores[:1]  # Complete the circle
        
        ax.plot(angles, scores, 'o-', linewidth=2, label=exp_name, color=colors[i])
        ax.fill(angles, scores, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_evaluation_results(results: Dict[str, Any], filepath: str) -> None:
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation results to {filepath}")


def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    p1 = "/home/huongtt/RWG/outputs/draft.txt"
    with open(p1, "r", encoding="utf-8") as f:
        related_work = f.read().strip()
    papers_idx = load_papers() 
    cited_papers = list(papers_idx.values()) 
    abstract = """
Wireless rechargeable sensor networks (WRSNs) employ mobile chargers (MCs) to replenish sensor energy, yet designing charging strategies that prolong network lifetime
while preserving coverage and connectivity remains challenging.
Offline (periodic) tours are inflexible to heterogeneous, time-
varying consumption, whereas online (on-demand) methods
must simultaneously decide the next charging location and
the delivered energy, and they often assume identical sensor
roles or rely on full-charge policies, leading to long waits and
premature MC returns. We address these limitations with a
generalized multi–mobile-charger framework that supports multi-
point charging, allowing an MC to replenish multiple sensors per
stop. We formulate cooperative control as a Decentralized Partially
Observable Semi-Markov Decision Process (Dec-POSMDP) to
handle partial observations and variable action durations. A
compact observation encoding—distribution descriptors with a
spatial heatmap—enables efficient reasoning and transfer across
network instances. We then learn decentralized policies via an
asynchronous multi-agent reinforcement learning solver based on
Independent Proximal Policy Optimization (IPPO). Experiments
on large-scale WRSNs show that the proposed method consistently
extends network lifetime while maintaining target coverage and
connectivity, and reduces charging delay and MC energy waste
compared with state-of-the-art baselines."
""" 
    evaluator = RelatedWorkEvaluator()
    results = evaluator.evaluate_related_work(related_work, abstract, cited_papers)
    output_dir = "/home/huongtt/RWG/outputs/evaluation"
    exp_name = "test"
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    for metric, data in results["metrics"].items():
        print(f"{metric:12s}: {data['score']}/5")
        print(f"  Reason: {data['reason'][:100]}...")
        print()
    
    print(f"Average Score: {results['average_score']:.2f}/5")
    print(f"Total Tokens Used: {results['total_tokens']}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{exp_name}_results.json")
    save_evaluation_results(results, results_path)
    
    # Create visualizations
    exp_results = {exp_name: results}
    
    # Bar chart
    bar_path = os.path.join(output_dir, f"{exp_name}_bar.png")
    plot_metrics_bar(exp_results, bar_path)
    
    # Radar chart
    radar_path = os.path.join(output_dir, f"{exp_name}_radar.png")
    plot_radar_chart(exp_results, radar_path)
    
    print(f"\n{'='*60}")
    print("Evaluation complete! Results and plots saved.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
