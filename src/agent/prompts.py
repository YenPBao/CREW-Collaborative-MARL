# =========================
# SELECTOR (graph traversal)
# =========================

SELECTOR_SYSTEM = "You are a research worker with excellent paper reading skills."
SELECTOR_USER = """
You are helping write the Related Work section of a scientific paper.
Your task: choose the NEXT paper section to read that adds the MOST NEW knowledge — not more of the same.

=== INPUTS ===
Citation graph: {graph}
Current paper (id=-1 = paper being written): {paper_details}
Cited papers: {cited_papers}
Citing papers: {citing_papers}
Your working memory (what YOU have read): {working_memory}
Your reading history (already read — do NOT re-read): {reading_history}
Team global knowledge (what ALL agents collectively know — treat as already covered): {global_knowledge}

=== MANDATORY REASONING (follow these steps in order) ===

STEP 1 — COVERAGE AUDIT
List the research themes already covered across YOUR memory AND the team global knowledge (max 7 bullets).
If both are empty, write "Coverage: empty".
Example:
  • [theme A] — in your memory
  • [theme B] — already in global knowledge (team covered)

STEP 2 — GAP IDENTIFICATION
List 2-3 themes NOT yet covered in either your memory OR global knowledge.
Think about: different method types, different modalities, different problem formulations,
different time periods, contrastive or opposing approaches.

STEP 3 — CANDIDATE SCORING
For each unread candidate paper (from cited_papers + citing_papers):
  - HIGH if its abstract addresses a gap from Step 2
  - LOW  if its themes are already covered in Step 1 (your memory OR global knowledge)

STEP 4 — SELECTION
Pick the highest-scoring HIGH candidate. If all are LOW, pick the one LEAST similar to global knowledge.

=== HARD RULES ===
- NEVER select a paper whose themes are >60% already covered in global knowledge — another agent read it.
- NEVER select a paper whose abstract is >60% overlapping with your own reading_history.
- If you have read 2+ papers on the same technique, do NOT pick a third — switch clusters.
- Section must exist EXACTLY in the paper's "structure" list. Available: "abstract", "related_work" only.
- Never repeat a (paper_id, section) pair from reading_history.
- Prefer "related_work" over "abstract" when available.

=== OUTPUT FORMAT (STRICT — one JSON only, no extra text) ===
{{"id": <paper_id>, "section": "<section_name>", "rationale": "<which gap this fills and why it is NOT already in global knowledge>"}}
"""


# =========================
# SEARCH (read + update memory; TEXT output)
# =========================

SEARCH_SYSTEM = "You are a research worker with excellent paper reading skills."
SEARCH_USER = """
You are updating a SHORT working memory (<= 4096 tokens) for writing the Related Work section.

=== INPUTS ===
Current paper (JSON, id=-1): {cur_paper}
Cited papers (JSON): {cited_paper}
You just read: section [{section}] of paper [{paper_id}]
Verbatim content:
{content}
Reading history (JSON): {reading_history}
Your previous memory:
{memory}
Current draft (already written):
{draft}
Global knowledge (all agents' shared pool):
{global_knowledge}

=== NOVELTY GATE (mandatory — do this BEFORE writing anything) ===

A) In ONE sentence, state the PRIMARY contribution of paper [{paper_id}] based on what you just read.

B) Check for overlap: does this primary contribution already appear in your memory, the draft,
   or the global knowledge?
   - OVERLAP (>60% similar topic/method): → go to MINIMAL MODE below.
   - NOVEL (genuinely new angle/method/dataset/paradigm): → go to FULL EXTRACTION below.

--- MINIMAL MODE (paper is largely redundant) ---
Extract ONE sentence only: the single most distinctive detail not found anywhere else, with [ID].
Do not elaborate. Keep memory addition under 30 words.

--- FULL EXTRACTION (paper is genuinely new) ---
Extract 2-4 sentences covering what is NEW:
  • The specific method, dataset, or finding that does NOT appear in memory/draft/global.
  • How it contrasts with OR complements existing approaches in your memory.
  • Do NOT copy/paraphrase text already in memory, draft, or global knowledge.

=== DIVERSITY RULES (apply in both modes) ===
- Each memory entry must represent a DISTINCT research direction. No two entries should
  describe the same technique with different wording.
- If memory already has 2+ entries on the same paradigm (e.g., attention, GAN, BERT-based),
  do NOT add another. Write "MEMORY_UNCHANGED: paradigm already covered." instead.
- Preferred new angles: unsupervised vs supervised, graph-based vs sequence-based,
  multi-modal vs single-modal, zero-shot vs fine-tuned, early work vs recent work.

=== CITATION RULES ===
- Use only IDs from the cited papers list: [ID] inline after the supported claim.
- Do NOT cite at paragraph ends. Do NOT invent IDs.

=== OUTPUT FORMAT (TEXT ONLY — no markdown, no extra lines) ===
MEMORY:
<updated memory text — your previous memory PLUS the new extracted content>
Reason: <one sentence explaining what gap this paper filled> [ID]
CITATIONS_USED: [ID1, ID2, ...]
"""


# =========================
# WRITE (final related work; JSON output)
# =========================

WRITE_SYSTEM = "You are a research worker with excellent paper writing skills."
WRITE_USER = """
Finalize the Related Work section of a scientific paper.

Target Context:
- Paper Abstract: {abstract}
- Cited Papers (JSON): {cited_paper}

Research Intelligence:
- Local Knowledge: {local_knowledge}
- Global Knowledge: {global_knowledge}

Collaboration:
- Existing Draft: {draft} (Current progress)
- Peer Feedbacks: {feedbacks} (Critiques to address)

STRICT INSTRUCTIONS:
1. Synthesis: Group studies by theme. Do NOT summarize papers in isolation.
2. Grounding: Use ONLY paper IDs from {{cited_paper}}. Never invent or use numeric indices like [1].
3. Traceability: Every claim must be supported by the provided knowledge. Remove untraceable citations.
4. Professionalism: Use a formal academic tone. Limit length to ~500-700 words.
5. Editing: If a draft exists, refine it focused on consistency and addressing feedbacks. Do not rewrite if the current version is good.
6. Topic Sentence Rule: Every paragraph must start with a clear thematic topic sentence.

Output Format:
Return ONLY a JSON object: {{ "related_work": "..." }}
"""


# =========================
# UPDATE (shared knowledge; TEXT output)
# =========================

UPDATE_SYSTEM = "You are a researcher worker with excellent knowledge synthesis skills, capable of organizing complex information into clear and logical insights."
UPDATE_USER = """
You must synthesize new insights from your Local Cache and integrate them into the existing Shared Knowledge Base to create a unified, high-quality reference for the entire agent network.

Current Shared Knowledge Base (text):
{shared_knowledge}

New information from Local Cache (may include MEMORY/CITATIONS_USED markers or JSON-wrapped text):
{local_cache}

If shared_knowledge is empty, you MUST first generate a professional **Thematic Outline** for the Related Work and then append synthesized content into the outline.

Quality Rules:
Non-Redundancy:
- Do not simply append. Merge overlaps, enrich existing points, and place new info logically.

High-Level Synthesis:
- Group related concepts, explain relationships, and highlight contrasts.
- Prefer integrative phrasing like: "While [ID1] focuses on X, [ID2] addresses Y by ..."

Conflict Resolution:
- If contradictions exist, resolve them when possible or document as a divergent view (with citations).

Citation Rules (MANDATORY, ANTI-SPAM):
- Preserve and include citations using ONLY paper IDs in square brackets [ID].
- You may ONLY cite an ID if the claim is explicitly supported by Local Cache content or the current Shared Knowledge.
- Do NOT invent new cited claims during synthesis.
- Transitional/organizational sentences may be uncited.

Anti-Dump (MANDATORY):
- Citations must appear within the same sentence right after the claim.
- Do NOT place large citation lists at paragraph ends.

Traceability Check (REQUIRED):
- Ensure every [ID] is traceable to the Local Cache or existing Shared Knowledge.
- Remove any untraceable citations.

Output Requirements (TEXT ONLY):
Return the updated Shared Knowledge as plain text. No JSON. No markdown.
"""


# =========================
# FEEDBACK (shared feedback; TEXT output)
# =========================

FEEDBACK_SYSTEM = "You are a researcher worker with excellent knowledge synthesis skills, specializing in providing constructive academic feedback."
FEEDBACK_USER = """
You must synthesize feedback insights from your Local Knowledge and integrate them into the existing Shared Feedback to create a unified, high-quality feedback reference for the Related Work draft.

Current Draft being reviewed:
{draft}

Your Local Knowledge (Author's Research Cache):
{local_knowledge}

Global Knowledge Base (shared context from all agents):
{global_knowledge}

Current Shared Feedback:
{current_feedback}

If current_feedback is empty, you MUST generate a comprehensive **Feedback Outline** evaluating the draft across ALL five dimensions below.

Evaluation Dimensions (score each 1–5, provide specific edits):

1. COVERAGE — Does the draft comprehensively cover all key areas?
   - Flag missing sub-fields, methods, or paradigms that should be cited.
   - Identify important papers from the knowledge base that are absent from the draft.

2. LOGIC — Is the logical flow and structure coherent?
   - Flag disordered paragraphs, weak transitions, or redundant repetition.
   - Suggest reordering or merging paragraphs where needed.

3. RELEVANCE — Is every sentence focused on the target paper's core topic?
   - Flag off-topic digressions or tangential citations.
   - Recommend removing or shortening sections that dilute focus.

4. DIVERSITY — Does the draft represent a BROAD range of approaches? (CRITICAL)
   - Check: are multiple distinct methodologies, paradigms, and viewpoints covered?
   - Flag if the draft over-relies on one cluster of papers or one technique.
   - Specifically recommend which different approaches (e.g., supervised vs. RL vs. graph-based;
     extractive vs. abstractive; single-model vs. multi-agent) are missing.
   - A score of 5 requires explicit comparison sentences between different approaches.

5. CITATION QUALITY — Are citations correctly anchored?
   - Flag claims without citations, citation dumping at paragraph ends, or invented IDs.
   - Recommend moving citations to be inline immediately after the supported claim.
   - Flag any citation that cannot be traced to the provided knowledge base.

Synthesis Rules:
- Non-Redundancy: merge overlapping critiques; do not repeat the same issue twice.
- Prioritization: lead with the highest-impact problems (Diversity and Coverage gaps first).
- Actionable: every critique must include a concrete fix or suggested sentence rewrite.
- Conflict Resolution: if prior feedback contradicts, resolve and keep the stricter version.

Output Requirements (TEXT ONLY):
Return the updated synthesized feedback as plain text. No JSON. No markdown.
"""


# =========================
# EVALUATION (strict reviewer; JSON output)
# =========================

EVALUATION_SYSTEM = "You are a strict paper reviewer. Be conservative with scores — do NOT give 5 unless the work is truly exceptional with no identifiable flaws. A fluent, well-written section is not automatically a 5; it must also be accurate and complete."

EVALUATION_USER = """
Please evaluate the Related Work section of the paper based on the abstracts of all the cited papers.

Important: Be strict and critical. Score 5 should be rare and only given when there are genuinely no weaknesses. Score 4 requires clear quality but must still note at least one concrete gap or issue. Most decent-but-imperfect sections should score 2–3. Never inflate scores just because the writing is fluent.

The scoring criteria are as follows:

Coverage
- Score 1: The 'Related Work' has limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.
- Score 2: The 'Related Work' covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.
- Score 3: The 'Related Work' is generally comprehensive in coverage but still misses a few key points that are not fully discussed.
- Score 4: The 'Related Work' covers most key areas of the topic comprehensively, with only very minor topics left out.
- Score 5: The 'Related Work' comprehensively covers all key and peripheral topics with no gaps — every research direction evident from the cited abstracts is discussed.

Logic
- Score 1: The 'Related Work' lacks logic, with no clear connections between sentences, making it difficult to understand the content.
- Score 2: The 'Related Work' has weak logical flow with some content arranged in a disordered or unreasonable manner.
- Score 3: The 'Related Work' has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated explanation.
- Score 4: The 'Related Work' has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts.
- Score 5: The 'Related Work' is tightly structured and logically clear, with all content arranged most reasonably, and transitions between adjacent sentences smooth without any redundancy.

Relevance
- Score 1: The content is outdated or unrelated to the field it purports to review, offering no alignment with the topic.
- Score 2: The 'Related Work' is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.
- Score 3: The 'Related Work' is generally on topic, despite a few unrelated details.
- Score 4: The 'Related Work' is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.
- Score 5: The 'Related Work' is exceptionally focused and entirely on topic — every claim is accurate relative to the cited abstracts and every sentence contributes meaningfully to the topic.

I will list the abstracts of all cited papers in JSON format like {{'id': id of the cited paper, 'abstract': abstract of the cited paper}}: {all_papers}
The content of the 'abstract' section of the current paper is: {abstract}
And the content of the 'Related Work' section to be evaluated is: {context}

Provide your reasoning for the score first, and then give the final score. Use JSON format for your response, like:
{{
  "Coverage": {{"reason": "The reason for your score", "score": 5}},
  "Logic": {{"reason": "The reason for your score", "score": 5}},
  "Relevance": {{"reason": "The reason for your score", "score": 5}},
}}
Be strict to the format and do not answer any other extra content!
"""

# =========================
# LONG CONTEXT (single-shot synthesis)
# =========================

LONG_CONTEXT_SYSTEM = "You are an expert researcher with excellent paper writing skills."
LONG_CONTEXT_USER = """
You are writing the Related Work section of a scientific paper. 

Target Context:
- Paper Abstract: {abstract}

Below are the abstracts of related papers (direct neighborhood in the citation graph). Your task is to synthesize these into a professional Related Work section.

Neighbor Abstracts:
{neighbor_abstracts}

STRICT INSTRUCTIONS:
1. Synthesis: Group studies by theme. Do NOT summarize papers in isolation.
2. Grounding: Use ONLY paper IDs from the provided list in square brackets [ID].
3. Professionalism: Use a formal academic tone. Limit length to ~600-800 words.
4. Topic Sentence Rule: Every paragraph must start with a clear thematic topic sentence.

Output Format:
Return ONLY a JSON object: {{ "related_work": "..." }}
"""


# =========================================================================
# TASK SWITCH: multi-document summarization (Multi-News)
# Khi RWG_PROMPTS=mds, override toàn bộ prompt bằng bản tóm tắt văn bản ở
# prompts_mds.py (giữ nguyên tên hằng/placeholder/định dạng — chỉ đổi task).
# =========================================================================
import os as _os
if _os.getenv("RWG_PROMPTS", "").lower() == "mds":
    from src.agent.prompts_mds import (
        SELECTOR_SYSTEM, SELECTOR_USER,
        SEARCH_SYSTEM, SEARCH_USER,
        WRITE_SYSTEM, WRITE_USER,
        UPDATE_SYSTEM, UPDATE_USER,
        FEEDBACK_SYSTEM, FEEDBACK_USER,
        EVALUATION_SYSTEM, EVALUATION_USER,
        LONG_CONTEXT_SYSTEM, LONG_CONTEXT_USER,
    )
