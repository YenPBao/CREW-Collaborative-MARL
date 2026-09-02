# src/agent/action.py
from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING, List, Optional, Set
from enum import IntEnum

import numpy as np
from src.agent.prompts import SEARCH_SYSTEM, SEARCH_USER, UPDATE_SYSTEM, UPDATE_USER, WRITE_SYSTEM, WRITE_USER, FEEDBACK_SYSTEM, FEEDBACK_USER,SELECTOR_SYSTEM,SELECTOR_USER
if TYPE_CHECKING:
    from src.rl_env.rwg_env import RWG_Enviroment
from src.core.utils import load_graph_out,load_papers,load_graph_in, extract_json
from src.core.config import example
from src.core.schemas import SearchResult
import json
class AgentAction(IntEnum):
    SEARCH = 0
    UPDATE = 1
    WRITE = 2
    FEEDBACK = 3

NUM_ACTIONS = len(AgentAction)
class ActSearch:
    def execute(self, env: "RWG_Enviroment", agent_id: str) -> str:
        title = env.paper_title
        abstract = env.abstract
        infor = title + abstract

        # Use pre-loaded data from environment
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

        # Only pass id + title + abstract — strip all other fields to cut prompt cost
        def _slim(p):
            return {"id": p.get("id"),
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", "")[:800]}

        cited_papers_slim  = [_slim(p) for p in cited_papers[:10]]
        citing_papers_slim = [_slim(p) for p in citing_papers[:10]]

        #select
        # Get working memory as string
        working_memory_str = json.dumps(local_memory.get("memory", {}),ensure_ascii=False)
        local_graph = {str(cur_idx): cited_ids} # All cited_ids

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

        # Randomized Start for First Search (User Request)
        if not history and len(cited_papers) > 0:
            import random
            random_paper = random.choice(cited_papers)
            target_id = random_paper.get("id")
            target_section = "abstract"
            sel_tokens = 0
            # Skip LLM generation
        else:
            sel_response, sel_tokens = llm.generate(selector_prompt, SELECTOR_SYSTEM)
            
            # Parse selector response using robust extract_json
            try:
                # Handle "End" response from selector
                if sel_response.strip().upper() == "END":
                    return f"Agent {agent_id} selector decided to end reading."
                
                selection = extract_json(sel_response)
                target_id = selection.get("id")
                target_section = selection.get("section")
                if target_id is None or target_section is None:
                    raise ValueError("Missing id or section")
            except Exception as e:
                # FALLBACK: If selector fails, pick the first unread citation if available
                print(f"    [Warning] Selector failed for {agent_id}: {e}. Using fallback.")
                target_id = None
                target_section = "abstract"
                
                # Check cited_papers first
                history_ids = {h.get("id") for h in history}
                for cp in cited_papers:
                    if cp.get("id") not in history_ids:
                        target_id = cp.get("id")
                        break
                
                if target_id is None:
                    # Try citing papers
                    for cp in citing_papers:
                         if cp.get("id") not in history_ids:
                            target_id = cp.get("id")
                            break
                            
                if target_id is None:
                    return f"Agent {agent_id} selector failed and no unread papers found in neighborhood. Stop search."
        
        # Normalize section name to match internal keys (e.g. "Related Work" -> "related_work")
        if isinstance(target_section, str):
            target_section = target_section.lower().strip().replace(" ", "_")

        # Validate target paper exists
        try:
            target_id = int(target_id)
        except (ValueError, TypeError):
            return f"Agent {agent_id} target paper ID '{target_id}' is invalid"

        if target_id not in papers:
            return f"Agent {agent_id} target paper {target_id} not found"
        
        target_paper = papers[target_id]
        # Validate section exists in paper structure
        if target_section not in target_paper.get("structure", []):
            return f"Agent {agent_id} section '{target_section}' not in paper {target_id} structure"
        
        #search
        # Get local memory as string for prompt
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
        
        # Parse search response - NOW PLAIN TEXT
        # We treat the entire response as memory
        search_result = {
            "memory": response, 
            "rationale": "Text response"
        }
        
        # Update env
        env.update_local_cache(agent_id, search_result)
        env.update_tokens(agent_id, total_tokens)
        # Track tokens for SEARCH action (L)
        env.action_token_usage[agent_id]["L"] += total_tokens
        
        # Update reading history to track what was read
        env.read_history[agent_id].append({
            "id": target_id,
            "section": target_section
        })
        
        with env._cited_lock:
            if not any(p.get("id") == target_id for p in env.cited_paper):
                env.cited_paper.append({
                    "id": target_id,
                    "title": target_paper.get("title", ""),
                    "abstract": target_paper.get("abstract", "")
                })
        
        return f"Agent {agent_id} successfully searched paper {target_id}, section '{target_section}' and updated local cache."
   
class ActWrite: 
    def execute(self, env: "RWG_Enviroment", agent_id: str) -> str:
        title = env.paper_title
        abstract = env.abstract
        infor = title + abstract
        local_memory = env.local_cache[agent_id]
        global_knowledge = env.global_knowledge
        draft = env.draft
        feedback = env.feedback  # Shared feedback string
        llm = env.llm
        cited_paper = env.cited_paper
        # Format local_memory as JSON string
        local_memory_str = json.dumps(local_memory, ensure_ascii=False) if isinstance(local_memory, dict) else str(local_memory)
        
        prompt = WRITE_USER.format(
            abstract=abstract[:2000], 
            cited_paper=json.dumps(cited_paper, ensure_ascii=False)[:3000],  # TODO: Get actual cited papers from environment
            local_knowledge=local_memory_str[:3000],
            global_knowledge=global_knowledge[:3000],
            draft=draft[:4000],
            feedbacks=feedback[:2000],  # Shared feedback string
            example=example 
        )
        response, tokens = llm.generate(prompt, WRITE_SYSTEM)
        print(f"    [ActWrite] Raw response length: {len(response)} chars")
        if len(response) < 100:
            print(f"    [ActWrite] Raw response: '{response}'")
        
        # Parse response using robust extract_json
        try:
            response_dict = extract_json(response)
            if not isinstance(response_dict, dict):
                response_dict = {"related_work": response}
        except Exception:
            # Fallback: treat entire response as related_work
            response_dict = {"related_work": response}
        
        env.update_draft(agent_id, response_dict)
        env.update_tokens(agent_id, tokens)
        # Track tokens for WRITE action (D)
        env.action_token_usage[agent_id]["D"] += tokens
        return f"Agent {agent_id} successfully wrote a draft."

class ActUpdate:
    def execute(self, env: "RWG_Enviroment", agent_id: str) -> str:
        local_cache = env.local_cache.get(agent_id, {})
        shared_knowledge = env.global_knowledge or ""
        llm = env.llm
        prompt = UPDATE_USER.format(
            shared_knowledge=shared_knowledge[:4000],
            local_cache=json.dumps(local_cache, ensure_ascii=False)[:4000]
        )
        
        
        response_text, tokens = llm.generate(prompt, UPDATE_SYSTEM)

        with env._global_lock:
            env.global_knowledge = response_text
        env.update_tokens(agent_id, tokens)
        env.action_token_usage[agent_id]["G"] += tokens
        return f"Agent {agent_id} successfully updated global knowledge."

class ActFeedback:
    """FEEDBACK action: similar to UPDATE, synthesize shared feedback from agent's perspective.
    Input: local knowledge + global knowledge + draft + current shared feedback
    Output: new shared feedback
    """
    def execute(self, env: "RWG_Enviroment", agent_id: str) -> str:
        draft = env.draft or ""
        local_knowledge = env.local_cache.get(agent_id, {})
        global_knowledge = env.global_knowledge or ""
        current_feedback = env.feedback or ""
        
        llm = env.llm
        local_knowledge_str = json.dumps(local_knowledge, ensure_ascii=False) if isinstance(local_knowledge, dict) else str(local_knowledge)
        
        prompt = FEEDBACK_USER.format(
            draft=draft[:4000], 
            local_knowledge=local_knowledge_str[:4000],
            global_knowledge=global_knowledge[:4000],
            current_feedback=current_feedback[:2000]
        )
        
        
        response_text, tokens = llm.generate(prompt, FEEDBACK_SYSTEM)

        with env._feedback_lock:
            env.feedback = response_text
        env.update_tokens(agent_id, tokens)
        env.action_token_usage[agent_id]["F"] += tokens
        return f"Agent {agent_id} successfully updated shared feedback."
ACTION_DISPATCHER: Dict[AgentAction, Any] = {
    AgentAction.SEARCH: ActSearch(),
    AgentAction.UPDATE: ActUpdate(),
    AgentAction.WRITE: ActWrite(),
    AgentAction.FEEDBACK: ActFeedback(),
}
