import os
import sys
import json
import time
import pandas as pd
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor

# Add root to sys.path
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import Generator
from src.evaluation.llm_base import RelatedWorkEvaluator
from src.agent.prompts import EVALUATION_USER, EVALUATION_SYSTEM, LONG_CONTEXT_USER, LONG_CONTEXT_SYSTEM
from src.core.utils import get_unique_cited_ids, load_papers, to_json
from src.agent.model import WorkerLLM

PROMPT_OVERHEAD = 800
RL_CKPT_PATH = None  # Set via --ckpt argument

# --- BFS Deep Recall Logic ---
def get_n_order_neighbors(adj, target_pid, max_depth=3):
    visited = {str(target_pid)}
    queue = [(str(target_pid), 0)]
    results = set()
    
    while queue:
        pid, depth = queue.pop(0)
        if depth > 0:
            results.add(pid)
        
        if depth < max_depth:
            neighbors = adj.get(str(pid), [])
            for n in neighbors:
                sn = str(n)
                if sn not in visited:
                    visited.add(sn)
                    queue.append((sn, depth + 1))
    return results

def calculate_recall(cited_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 0.0
    matches = set(cited_ids).intersection(set(ground_truth_ids))
    return len(matches) / len(ground_truth_ids)

# --- Custom Evaluator to follow prompts.py exactly ---
class CustomEvaluator(RelatedWorkEvaluator):
    def evaluate(self, draft, abstract, all_papers_list):
        prompt = EVALUATION_USER.format(
            all_papers=json.dumps(all_papers_list[:50]), # Limit context
            abstract=abstract,
            context=draft
        )
        response, tokens = self.llm.generate(prompt, system_prompt=EVALUATION_SYSTEM)
        try:
            # Clean JSON - more robust version
            json_str = response.strip()
            # Remove any non-JSON prefix if it exists
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0].strip()
            elif "{" in json_str:
                # Fallback: extract everything between first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}")
                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
            
            # Remove potential JS comments or trailing commas that some LLMs include
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*\]", "]", json_str)
            
            eval_data = json.loads(json_str)
            # Normalize to include all 4 metrics even if some are missing
            metrics = ["Coverage", "Logic", "Relevance", "Citation"]
            normalized = {}
            for m in metrics:
                # Handle case where metric might be missing or in different case
                val = next((v for k, v in eval_data.items() if k.lower() == m.lower()), {"score": 0, "reason": "N/A"})
                if isinstance(val, (int, float)):
                    normalized[m] = {"score": val, "reason": "N/A"}
                else:
                    normalized[m] = val
            return normalized
        except Exception as e:
            print(f"Error parsing eval: {e}. Raw: {response[:100]}")
            return {}

# --- Main Pipeline ---
def run_and_eval(limit=None, target_id=None, overwrite=False):
    output_base = ROOT / "result"
    output_base.mkdir(exist_ok=True)
    draft_dir = output_base / "drafts"
    log_dir = output_base / "logs"
    draft_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    execution_log_f = output_base / "execution_log.txt"
    
    def log_and_print(msg):
        print(msg)
        with open(execution_log_f, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log_and_print("="*50)
    log_and_print("STARTING COMPREHENSIVE EVALUATION PIPELINE")
    log_and_print("="*50)

    # Load high-density papers
    metadata_f = ROOT / "target_papers_metadata.json"
    if not metadata_f.exists():
        print(f"Metadata not found at {metadata_f}. Please run paper selection first.")
        return
    with open(metadata_f, 'r', encoding='utf-8') as f:
        target_papers = json.load(f)

    # Load paper groups for logging
    group_f = ROOT / "neighborhood_selected_papers_v6.json"
    id_to_group = {}
    if group_f.exists():
        with open(group_f, 'r', encoding='utf-8') as f:
            group_data = json.load(f)
            for p in group_data.get('high_density', []): id_to_group[str(p['id'])] = "High Density"
            for p in group_data.get('low_density', []): id_to_group[str(p['id'])] = "Low Density"

    if target_id:
        target_papers = [p for p in target_papers if str(p['id']) == str(target_id)]
    if limit:
        if limit < 0:
            target_papers = target_papers[limit:]
        else:
            target_papers = target_papers[:limit]
        
    log_and_print(f"Executing on {len(target_papers)} papers.")

    # Load graph for recall
    log_and_print("Loading graph for recall and paper database...")
    adj_f = ROOT / 'data2/citation_adj_out.json'
    with open(adj_f, 'r') as f:
        graph = json.load(f)
        
    # Load paper database (subset mode)
    log_and_print("Loading paper database (subset mode)...")
    relevant_pids = set()
    for p in target_papers:
        relevant_pids.add(int(p['id']))
        for ref in p.get('referenced', []):
            rid = int(ref['id'])
            relevant_pids.add(rid)
            # Add level-2 neighbors (references of references)
            for r2_id in ref.get('referenced_ids', []):
                relevant_pids.add(int(r2_id))
    
    all_papers_db = load_papers(limit_ids=relevant_pids)
    log_and_print(f"Loaded {len(all_papers_db)} papers into evaluation database (including level-2 neighbors).")

    results = []

    for paper in target_papers:
        pid = paper['id']
        title = paper['title']
        abstract = paper['abstract']
        group_name = id_to_group.get(str(pid), "Unknown")
        log_and_print(f"\n>>> Processing Paper {pid} [{group_name}]: {title[:100]}...")
        ground_truth_ids = [str(r['id']) for r in paper.get('referenced', [])]

        # Branch IDs for Deep Recall
        n1_ids = get_n_order_neighbors(graph, pid, max_depth=1)
        n3_ids = get_n_order_neighbors(graph, pid, max_depth=3)
        print(f"    Graph Info: N1={len(n1_ids)}, N3={len(n3_ids)}")

        # 1. Run RL and Baseline scenarios (S2-S5) to get token baseline
        token_baseline = 0
        scenarios = [
            {"name": "S2_RL_1A", "agents": 1, "steps": 15, "heuristic": False, "model": "openrouter:qwen/qwen-2.5-72b-instruct"},
            {"name": "S3_GPT4o_RAG", "agents": 1, "steps": 5, "heuristic": True, "model": "openai:gpt-4o"}, 
            {"name": "S4_RL_2A", "agents": 2, "steps": 15, "heuristic": False, "model": "openrouter:qwen/qwen-2.5-72b-instruct"},
            {"name": "S5_RL_3A", "agents": 3, "steps": 15, "heuristic": False, "model": "openrouter:qwen/qwen-2.5-72b-instruct"},
            {"name": "S6_RL_4A", "agents": 4, "steps": 15, "heuristic": False, "model": "openrouter:qwen/qwen-2.5-72b-instruct"},
        ]
        
        PROMPT_OVERHEAD = 800 # Tokens to subtract per call
        
        paper_results = {}

        for scen in scenarios:
            name = scen['name']
            draft_path = draft_dir / f"P{pid}_{name}.txt"
            
            if not overwrite and draft_path.exists() and draft_path.stat().st_size > 0:
                log_and_print(f"    Skipping generation for {name}, loading existing draft...")
                with open(draft_path, "r", encoding="utf-8") as f:
                    draft = f.read()
                paper_results[name] = {
                    "draft": draft,
                    "draft_path": str(draft_path),
                    "tokens_in": 0, 
                    "tokens_out": 0,
                    "total_tokens": 0,
                    "num_calls": 0
                }
                continue

            log_and_print(f"    Running Scenario {name} using {scen['model']}...")
            cur_log_dir = log_dir / f"P{pid}_{name}"
            gen = Generator(
                agents=scen['agents'], 
                steps=scen['steps'], 
                heuristic=scen['heuristic'],
                model_name=scen['model'],
                ckpt_path=RL_CKPT_PATH if not scen['heuristic'] else None,
                output_dir=str(cur_log_dir),
                preloaded_papers=all_papers_db
            )
            res = gen.generate(title, abstract, paper_id=pid)
            
            draft = res.get('draft', '')
            
            if not draft.strip():
                log_and_print(f"    [Warning] In-memory draft for {name} is empty. Searching log dir...")
                md_files = list(cur_log_dir.glob("related_work_*.md"))
                if md_files:
                    latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
                    log_and_print(f"    [Fallback] Found potential draft: {latest_md.name}")
                    with open(latest_md, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "## Draft" in content:
                            draft = content.split("## Draft")[-1].split("---")[0].strip()
                        if draft == "*No draft*": draft = ""

            usage = res.get('token_usage', {})
            raw_in = usage.get('prompt_tokens', 0)
            n_calls = usage.get('num_calls', 0)
            adj_in = max(0, raw_in - n_calls * PROMPT_OVERHEAD)
            tokens_out = usage.get('completion_tokens', 0)
            adj_total = adj_in + tokens_out
            
            token_baseline = max(token_baseline, adj_total)
            with open(draft_path, "w", encoding="utf-8") as f:
                f.write(draft)
            
            paper_results[name] = {
                "draft": draft,
                "draft_path": str(draft_path),
                "tokens_in": adj_in,
                "tokens_out": tokens_out,
                "total_tokens": adj_total,
                "num_calls": n_calls
            }

        # 2. Run S1 with Token Balancing (Early Stop Mode)
        log_and_print(f"    Running Scenario S1_Select_Read_and_Write with Token Balancing (Target: {token_baseline}) using Qwen...")
        s1_draft_path = draft_dir / f"P{pid}_S1_Select_Read_and_Write.txt"
        
        # Scenario S1: Start with high step count (e.g. 60) and stop at token_baseline
        # Per user request, S1 does NOT use the 800-token prompt overhead deduction
        gen_s1 = Generator(
            agents=1, 
            steps=60, 
            heuristic=True,
            model_name="openrouter:qwen/qwen-2.5-72b-instruct",
            output_dir=str(log_dir / f"P{pid}_S1_balancing"),
            preloaded_papers=all_papers_db
        )
        res_s1 = gen_s1.generate(title, abstract, paper_id=pid, token_limit=token_baseline, prompt_overhead=0)
        s1_draft = res_s1.get('draft', '')
        
        usage_s1 = res_s1.get('token_usage', {})
        s1_prompt_tokens = usage_s1.get('prompt_tokens', 0)
        s1_completion_tokens = usage_s1.get('completion_tokens', 0)
        s1_n_calls = usage_s1.get('num_calls', 0)
        
        # No deduction for S1
        s1_adj_in = s1_prompt_tokens
        s1_total_tokens = s1_adj_in + s1_completion_tokens
        
        with open(s1_draft_path, "w", encoding="utf-8") as f:
            f.write(s1_draft)
            
        paper_results["S1_Select_Read_and_Write"] = {
            "draft": s1_draft,
            "draft_path": str(s1_draft_path),
            "tokens_in": s1_adj_in,
            "tokens_out": s1_completion_tokens,
            "total_tokens": s1_total_tokens,
            "num_calls": s1_n_calls
        }
        
        # 2.5 Add S0_Human Scenario (Original Human Related Work)
        log_and_print("    Extracting S0_Human (Original Related Work)...")
        human_rw = all_papers_db.get(str(pid), {}).get("related_work", "")
        if human_rw:
            paper_results["S0_Human"] = {
                "draft": human_rw,
                "draft_path": "N/A (Original)",
                "tokens_in": 0,
                "tokens_out": 0,
                "total_tokens": 0,
                "num_calls": 0
            }
        else:
            log_and_print(f"    [Warning] No original Related Work found for paper {pid}")
            
        # 2.7 Scenario S7: GPT-4o Long Context (Single-Shot Baseline)
        log_and_print("    Running Scenario S7_GPT4o_LongContext...")
        s7_draft_path = draft_dir / f"P{pid}_S7_GPT4o_LongContext.txt"
        
        if not overwrite and s7_draft_path.exists():
            log_and_print("    Scenario S7 already exists. Skipping generation.")
            with open(s7_draft_path, "r", encoding="utf-8") as f:
                s7_draft = f.read()
            paper_results["S7_GPT4o_LongContext"] = {
                "draft": s7_draft,
                "draft_path": str(s7_draft_path),
                "tokens_in": 0, 
                "tokens_out": 0,
                "total_tokens": 0,
                "num_calls": 0
            }
        else:
            neighbor_abstracts = ""
            # Gather abstracts from ground_truth_ids (direct citations)
            for ref_id in ground_truth_ids:
                ref_paper = all_papers_db.get(str(ref_id), {})
                if ref_paper:
                    neighbor_abstracts += f"Paper ID: [{ref_id}]\nAbstract: {ref_paper.get('abstract', 'N/A')}\n\n"
            
            llm_s7 = WorkerLLM(model_name="openai:gpt-4o")
            prompt_s7 = LONG_CONTEXT_USER.format(abstract=abstract, neighbor_abstracts=neighbor_abstracts)
            response_s7, tokens_s7 = llm_s7.generate(prompt_s7, system_prompt=LONG_CONTEXT_SYSTEM)
            
            # Parse JSON safely
            try:
                if "{" in response_s7 and "}" in response_s7:
                    s7_data = to_json(response_s7)
                    s7_draft = s7_data.get("related_work", response_s7)
                else:
                    s7_draft = response_s7
            except:
                s7_draft = response_s7
            
            if not (s7_draft or "").strip(): s7_draft = response_s7
            
            with open(s7_draft_path, "w", encoding="utf-8") as f:
                f.write(s7_draft)
            
            pi = tokens_s7.get("prompt_tokens", 0)
            ci = tokens_s7.get("completion_tokens", 0)
            paper_results["S7_GPT4o_LongContext"] = {
                "draft": s7_draft,
                "draft_path": str(s7_draft_path),
                "tokens_in": pi,
                "tokens_out": ci,
                "total_tokens": pi + ci,
                "num_calls": 1
            }

        # 3. Evaluation Phase
        eval_models = [
            "openai:gpt-4o",
            "openrouter:qwen/qwen-2.5-7b-instruct",
            "openrouter:meta-llama/llama-3.1-8b-instruct:free",
            "openrouter:anthropic/claude-3.5-haiku"
        ]
        
        for name, data in paper_results.items():
            draft = data['draft']
            if not draft.strip():
                log_and_print(f"    [Skip] Draft for {name} is empty, skipping evaluation.")
                continue

            log_and_print(f"    Evaluating {paper['id']} {name}...")
            cited_ids = get_unique_cited_ids(draft)
            rec1 = calculate_recall(cited_ids, n1_ids)
            rec3 = calculate_recall(cited_ids, n3_ids)
            
            # Metrics to track
            metrics = ["Coverage", "Logic", "Relevance", "Citation"]
            metric_sums = {m: 0 for m in metrics}
            eval_count = 0
            
            for emodel in eval_models:
                log_and_print(f"      - Using evaluator: {emodel}...")
                evaluator = CustomEvaluator(model_name=emodel)
                
                # Build Smart Context: real abstracts for cited papers + some neighbors
                smart_context = []
                added_ids = set()
                
                # 1. Add papers actually cited in the draft
                for cid in cited_ids:
                    p_info = all_papers_db.get(str(cid))
                    if p_info:
                        smart_context.append({"id": cid, "abstract": p_info.get("abstract", "")[:800]})
                        added_ids.add(str(cid))
                
                # 2. Add neighbors that were NOT cited to test Coverage
                for nid in list(n1_ids):
                    if str(nid) not in added_ids and len(smart_context) < 25:
                        p_info = all_papers_db.get(str(nid))
                        if p_info:
                            smart_context.append({"id": nid, "abstract": p_info.get("abstract", "")[:500]})
                            added_ids.add(str(nid))
                            
                scores = evaluator.evaluate(draft, abstract, smart_context)
                if scores:
                    for m in metrics:
                        # Extract score, ensure it's a number
                        s_data = scores.get(m, {})
                        val = s_data.get("score", 0) if isinstance(s_data, dict) else 0
                        metric_sums[m] += val
                    eval_count += 1
            
            avg_metrics = {m: (metric_sums[m] / eval_count if eval_count > 0 else 0) for m in metrics}
            final_avg = sum(avg_metrics.values()) / 4 if metrics else 0
            
            row = {
                "Paper ID": pid,
                "Density_Group": group_name,
                "Scenario": name,
                "Recall_1": rec1,
                "Recall_3": rec3,
                "Token_In": data['tokens_in'],
                "Token_Out": data['tokens_out'],
                "Total_Tokens": data['total_tokens'],
                "Avg_Score": final_avg,
                "Coverage": avg_metrics["Coverage"],
                "Logic": avg_metrics["Logic"],
                "Relevance": avg_metrics["Relevance"],
                "Citation": avg_metrics["Citation"],
                "Draft_Path": data['draft_path'],
                "Evaluators": eval_count
            }
            results.append(row)
            pd.DataFrame(results).to_csv(ROOT / "result/final_scores.csv", index=False)
            log_and_print(f"    [Done] Avg Score: {final_avg:.2f} | R1: {rec1:.2f}")

    log_and_print("\n" + "="*50)
    log_and_print("COMPREHENSIVE PIPELINE FINISHED!")
    log_and_print(f"Results saved to result/final_scores.csv")
    log_and_print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target_id", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to RL checkpoint (e.g. ./checkpoints/self_supervised_openai/2026-03-21-13-42/latest)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated drafts")
    args = parser.parse_args()

    if args.ckpt:
        RL_CKPT_PATH = args.ckpt

    run_and_eval(limit=args.limit, target_id=args.target_id, overwrite=args.overwrite)
