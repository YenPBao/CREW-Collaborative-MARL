# src/agent/action_no_feedback.py
from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING, List, Optional, Set
from enum import IntEnum
import json
import numpy as np
from src.agent.prompts import SEARCH_SYSTEM, SEARCH_USER, UPDATE_SYSTEM, UPDATE_USER, WRITE_SYSTEM, WRITE_USER, SELECTOR_SYSTEM, SELECTOR_USER
if TYPE_CHECKING:
    from src.rl_env.rwg_env_no_feedback import RWG_Environment_NoFeedback
from src.core.utils import load_graph_out, load_papers, load_graph_in, extract_json
from src.core.config import example
from src.core.schemas import SearchResult

class AgentAction(IntEnum):
    SEARCH = 0
    UPDATE = 1
    WRITE = 2

NUM_ACTIONS = len(AgentAction)

class ActSearch:
    def execute(self, env: "RWG_Environment_NoFeedback", agent_id: str) -> str:
        title = env.paper_title
        abstract = env.abstract
        
        graph_out = env.citation_adj_out
        graph_in = env.citation_adj_in
        papers = env.papers

        local_memory = env.local_cache[agent_id]
        history = env.read_history[agent_id]
         
        if not history:
            cur_idx = -1
        else:
            cur_idx = history[-1].get("id", -1)

        cur_paper_detailed = papers.get(cur_idx, {})
        cited_ids = graph_out.get(cur_idx, [])
        citing_ids = graph_in.get(cur_idx, []) 

        cited_papers = [papers[pid] for pid in cited_ids if pid in papers]
        citing_papers = [papers[pid] for pid in citing_ids if pid in papers]
        llm = env.llm

        def _slim(p):
            return {"id": p.get("id"),
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", "")[:800]}

        cited_papers_slim  = [_slim(p) for p in cited_papers[:10]]
        citing_papers_slim = [_slim(p) for p in citing_papers[:10]]

        working_memory_str = json.dumps(local_memory.get("memory", {}), ensure_ascii=False)
        local_graph = {str(cur_idx): cited_ids}
        global_knowledge_str = (env.global_knowledge or "")[:2000]

        selector_prompt = SELECTOR_USER.format(
            graph=json.dumps(local_graph)[:1000],
            paper_details=json.dumps(cur_paper_detailed).strip()[:2000],
            cited_papers=json.dumps(cited_papers_slim),
            citing_papers=json.dumps(citing_papers_slim),
            working_memory=working_memory_str[:2000],
            reading_history=json.dumps(history)[-1000:],
            global_knowledge=global_knowledge_str,
        )

        if not history and len(cited_papers) > 0:
            import random
            random_paper = random.choice(cited_papers)
            target_id = random_paper.get("id")
            target_section = "abstract"
            sel_tokens = 0
        else:
            sel_response, sel_tokens = llm.generate(selector_prompt, SELECTOR_SYSTEM)
            try:
                if sel_response.strip().upper() == "END":
                    return f"Agent {agent_id} selector decided to end reading."
                selection = extract_json(sel_response)
                target_id = selection.get("id")
                target_section = selection.get("section")
            except Exception as e:
                target_id = None
                target_section = "abstract"
                history_ids = {h.get("id") for h in history}
                for cp in cited_papers:
                    if cp.get("id") not in history_ids:
                        target_id = cp.get("id")
                        break
                if target_id is None:
                    for cp in citing_papers:
                         if cp.get("id") not in history_ids:
                            target_id = cp.get("id")
                            break
                if target_id is None:
                    return f"Agent {agent_id} selector failed and no unread papers found. Stop search."
        
        if isinstance(target_section, str):
            target_section = target_section.lower().strip().replace(" ", "_")

        try:
            target_id = int(target_id)
        except (ValueError, TypeError):
            return f"Agent {agent_id} target paper ID '{target_id}' is invalid"

        if target_id not in papers:
            return f"Agent {agent_id} target paper {target_id} not found"
        
        target_paper = papers[target_id]
        if target_section not in target_paper.get("structure", []):
            return f"Agent {agent_id} section '{target_section}' not in paper {target_id} structure"
        
        memory_str = json.dumps(local_memory.get("memory", {}), ensure_ascii=False)
        content = target_paper.get(target_section, "")
        target_title = target_paper.get("title", "")

        search_prompt = SEARCH_USER.format(
            cur_paper=json.dumps(cur_paper_detailed).strip()[:2000],
            cited_paper=json.dumps(cited_papers_slim),
            target_title=target_title[:500],
            section=target_section,
            paper_id=target_id,
            reading_history=json.dumps(history)[-1000:],
            content=json.dumps(content)[:8000],
            memory=memory_str[:2000],
            draft=env.draft[:2000],
            global_knowledge=env.global_knowledge[:2000],
        )
        response, se_tokens = llm.generate(search_prompt, SEARCH_SYSTEM)
        total_tokens = sel_tokens + se_tokens
        
        search_result = {"memory": response, "rationale": "Text response"}
        env.update_local_cache(agent_id, search_result)
        env.update_tokens(agent_id, total_tokens)
        env.action_token_usage[agent_id]["L"] += total_tokens
        env.read_history[agent_id].append({"id": target_id, "section": target_section})
        
        with env._cited_lock:
            if not any(p.get("id") == target_id for p in env.cited_paper):
                env.cited_paper.append({
                    "id": target_id,
                    "title": target_paper.get("title", ""),
                    "abstract": target_paper.get("abstract", "")
                })
        
        return f"Agent {agent_id} successfully searched paper {target_id}, section '{target_section}'."

class ActWrite: 
    def execute(self, env: "RWG_Environment_NoFeedback", agent_id: str) -> str:
        abstract = env.abstract
        local_memory = env.local_cache[agent_id]
        global_knowledge = env.global_knowledge
        draft = env.draft
        llm = env.llm
        cited_paper = env.cited_paper
        local_memory_str = json.dumps(local_memory, ensure_ascii=False) if isinstance(local_memory, dict) else str(local_memory)
        
        prompt = WRITE_USER.format(
            abstract=abstract[:2000], 
            cited_paper=json.dumps(cited_paper, ensure_ascii=False)[:3000],
            local_knowledge=local_memory_str[:3000],
            global_knowledge=global_knowledge[:3000],
            draft=draft[:4000],
            feedbacks="",  # Feedback is empty in ablation study
            example=example 
        )
        response, tokens = llm.generate(prompt, WRITE_SYSTEM)
        try:
            response_dict = extract_json(response)
            if not isinstance(response_dict, dict):
                response_dict = {"related_work": response}
        except Exception:
            response_dict = {"related_work": response}
        
        env.update_draft(agent_id, response_dict)
        env.update_tokens(agent_id, tokens)
        env.action_token_usage[agent_id]["D"] += tokens
        return f"Agent {agent_id} successfully wrote a draft."

class ActUpdate:
    def execute(self, env: "RWG_Environment_NoFeedback", agent_id: str) -> str:
        local_cache = env.local_cache.get(agent_id, {})
        shared_knowledge = env.global_knowledge or ""
        llm = env.llm
        prompt = UPDATE_USER.format(
            shared_knowledge=shared_knowledge[:4000],
            local_cache=json.dumps(local_cache, ensure_ascii=False)[:4000]
        )
        response_text, tokens = llm.generate(prompt, UPDATE_SYSTEM)
        env.global_knowledge = response_text
        env.update_tokens(agent_id, tokens)
        env.action_token_usage[agent_id]["G"] += tokens
        return f"Agent {agent_id} successfully updated global knowledge."

ACTION_DISPATCHER: Dict[AgentAction, Any] = {
    AgentAction.SEARCH: ActSearch(),
    AgentAction.UPDATE: ActUpdate(),
    AgentAction.WRITE: ActWrite(),
}
