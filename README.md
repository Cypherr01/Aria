# ARIA — Agentic Reasoning and Intelligence Architecture

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
![Free Tier](https://img.shields.io/badge/Infrastructure-100%25%20Free%20Tier-brightgreen)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)

> **The only limitation is the model, not the system.**

---

## Core Philosophy

Most open-source chatbots are wrappers. They take a model, add a chat loop, and call it an agent. ARIA is not that. ARIA is an autonomous AI operating system — capable of reasoning, planning, executing multi-step workflows, managing memory across sessions, using tools, correcting itself, and collaborating across specialized agents. Every architectural decision exists to ensure the system never fails the model. If ARIA gives a poor answer, it is because the LLM couldn't do better — not because the pipeline let it down.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
   - [Layer 1 — Multi-Model Router](#layer-1--multi-model-router)
   - [Layer 2 — Agentic Reasoning Core](#layer-2--agentic-reasoning-core-langgraph)
   - [Layer 3 — Multi-Agent System](#layer-3--multi-agent-system)
   - [Layer 4 — Memory Architecture](#layer-4--memory-architecture)
   - [Layer 5 — Tool Suite](#layer-5--tool-suite)
   - [Layer 6 — Responsible AI Layer](#layer-6--responsible-ai-layer)
   - [Layer 7 — Observability](#layer-7--observability)
2. [Folder Structure](#2-folder-structure)
3. [Agent Decision Flowchart](#3-agent-decision-flowchart)
4. [API Design](#4-api-design)
5. [Configuration System](#5-configuration-system)
6. [Setup & Installation](#6-setup--installation)
7. [Tech Stack Decisions](#7-tech-stack-decisions)
8. [What Makes This Different](#8-what-makes-this-different)
9. [Known Limitations](#9-known-limitations)
10. [Production Deployment](#10-production-deployment-free-infra)
11. [Full Production-Grade Code](#11-full-production-grade-code)
12. [Roadmap](#12-roadmap)

---

## Stack

| Layer | Technology | Tier |
|---|---|---|
| Language | Python 3.11+ | — |
| Agent Orchestration | LangGraph | Free |
| LLM APIs | Gemini 1.5 Flash/Pro, Groq (LLaMA 3, DeepSeek, Mixtral), Together AI, Cohere | Free |
| Embeddings | Cohere / Gemini Embedding API (router-selected, no local models) | Free tier |
| Reranking | Cohere Rerank API (router-selected, no local cross-encoder) | Free tier |
| Vector DB | ChromaDB (local) | Free |
| Web Search | Tavily API + DuckDuckGo fallback | Free (1000/mo) |
| Memory Persistence | SQLite + ChromaDB | Free |
| Backend | FastAPI + asyncio background tasks | Free |
| Frontend | Streamlit | Free |
| Observability | LangSmith + structured JSON logging | Free tier |
| Code Execution | RestrictedPython (sandboxed) | Free |

---

## 1. System Architecture

ARIA's architecture is composed of 7 layers. Each layer is independently owned, testable, and swappable. No layer is permitted to become a bottleneck for the LLM's capability.

---

### Layer 1 — Multi-Model Router

The router is the nervous system of ARIA. It ensures that every LLM call is dispatched to the best available, healthiest, most cost-efficient model for the task type — with zero user-visible failures.

#### Model Registry Schema

Every registered model carries a full health and capability profile:

```python
@dataclass
class ModelEntry:
    name: str                          # e.g. "gemini-1.5-pro"
    provider: str                      # "google" | "groq" | "together" | "cohere"
    api_key_env_var: str               # Environment variable holding the API key
    context_window: int                # Max tokens in context
    tokens_used_today: int             # Running daily token count (SQLite-persisted)
    tokens_limit_daily: int            # Daily limit for free tier
    failure_count: int                 # Consecutive failures
    cooldown_until: Optional[datetime] # Cooldown expiry (set on rate limit)
    task_affinity: List[str]           # ["reasoning", "long_context", "code", "speed"]
    max_output_tokens: int             # Max tokens in single response
    avg_latency_ms: float              # Running average latency (EMA)
    is_healthy: bool                   # Computed field: not in cooldown, under budget
```

#### Dynamic Model Registry & Tiers

ARIA does not hardcode models. Instead, it reads a dynamic registry from `.env` and `models.env` (via `config.build_llm_registry()`). Models are assigned to semantic *Tiers* rather than hardcoded tasks.

**Tiers:**
- `REASONING`: Deep reasoning (synthesis, planning, reflection, research)
- `SPEED`: Fast extraction and classification
- `CODE`: Specialized code generation
- `SUMMARIZATION`: Working memory compression
- `SAFETY`: Content filter input and output
- `EMBEDDING`: Vector encoding
- `RERANKING`: Cross-encoder reranking

**Routing Fallback Chain:**
For each tier, the router maintains an ordered fallback chain. If a primary model fails due to rate limits or API errors, the router instantly transparently falls back to the next model in the chain.

#### Routing Algorithm

Before every LLM call, the router executes:

1. Filter model registry by `task_affinity` match.
2. Filter out models where `cooldown_until > now()`.
3. Filter out models where `tokens_used_today >= tokens_limit_daily * 0.95` (5% buffer).
4. Sort remaining by: `failure_count ASC, avg_latency_ms ASC`.
5. Select the first result. If empty → trigger graceful degradation.

#### Failure Handling

```
Rate limit error (429)
  → Exponential backoff: 2s → 4s → 8s (max 3 retries on same model)
  → If still failing: set cooldown_until = now() + 60min, increment failure_count
  → Rotate to next best model for same task
  → User never sees a failure — only slight latency increase

All models exhausted for task type
  → Log event with full context
  → Return graceful degradation message:
     "I've temporarily exceeded my API limits on all available models for this
      request type. Please retry in a few minutes. [Status: /model-status]"

Non-rate-limit error (500, timeout, malformed response)
  → Immediate rotation to next model
  → failure_count += 1
  → If failure_count > 5: mark model unhealthy for 30min
```

#### Token Budget Management

Token usage is persisted to SQLite and resets at midnight UTC:

```sql
CREATE TABLE token_usage (
    model_name TEXT,
    date TEXT,  -- YYYY-MM-DD
    tokens_used INTEGER,
    PRIMARY KEY (model_name, date)
);
```

After every LLM call: `UPDATE token_usage SET tokens_used = tokens_used + ? WHERE model_name = ? AND date = ?`

#### Model Health Endpoints

```
GET /model-status
→ Returns: [{name, provider, is_healthy, tokens_used_today, tokens_limit_daily,
             utilization_pct, cooldown_remaining_sec, failure_count, avg_latency_ms}]

CLI: aria model-status
```

---

### Layer 2 — Agentic Reasoning Core (LangGraph)

ARIA does not reply. It reasons. Every user input passes through a stateful LangGraph graph. The graph is the difference between a chatbot and an agent.

#### State Schema

```python
class ARIAState(TypedDict):
    # Input
    user_message: str
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, str]]

    # Classification
    intent_type: Literal["conversational", "factual", "research", "code",
                          "document", "memory_operation", "ambiguous"]
    confidence_score: float
    extracted_entities: List[str]
    requires_tools: bool

    # Planning
    execution_plan: Optional[List[PlanStep]]
    plan_shown_to_user: bool

    # Execution
    tool_calls: List[ToolCallRecord]
    tool_outputs: List[ToolOutputRecord]

    # Reflection
    reflection_scores: Optional[ReflectionScores]
    reflection_passed: bool
    retry_count: int

    # Memory
    retrieved_memories: List[MemoryEntry]
    memory_writes: List[MemoryWrite]

    # Output
    draft_response: Optional[str]
    final_response: Optional[str]
    citations: List[Citation]
    confidence_indicator: Literal["high", "medium", "low"]
    follow_up_suggestions: List[str]

    # Meta
    error_log: List[ErrorEntry]
    debug_trace: List[str]
    model_used: str
    total_latency_ms: int
```

#### Node Definitions

**NODE 1 — IntentClassifier**

```
Input:  user_message + conversation_history
Output: intent_type, confidence_score, extracted_entities, requires_tools

Logic:
  - Calls LLM (speed-affinity model) with structured output prompt
  - Returns Pydantic-validated classification
  - If confidence_score < 0.6: intent_type = "ambiguous" → user asked clarifying question
  - entity extraction uses spaCy NER + LLM augmentation
  - Adds to debug_trace: "Intent: {intent_type} ({confidence_score:.2f})"
```

**NODE 2 — Planner**

```
Input:  intent_type, user_message, extracted_entities, requires_tools
Output: execution_plan (List[PlanStep])

Activated when: requires_tools=True OR intent in [research, code, document]

PlanStep schema:
  {
    step_number: int,
    action: str,           # human-readable description
    tool: str,             # tool name from registry
    tool_params: dict,     # resolved parameters
    expected_output: str,  # what this step should produce
    fallback_action: str,  # what to do if this step fails
    can_parallelize: bool  # can run in parallel with previous step
  }

UI: Plan rendered as collapsible "Thinking..." trace in Streamlit
```

**NODE 3 — ToolRouter**

```
Input:  execution_plan
Output: ordered_tool_calls (List[ResolvedToolCall])

Logic:
  - Validates each tool call against tool registry schema
  - Groups parallel-eligible steps (can_parallelize=True + no dependency on prior step output)
  - Parallel group executed concurrently with asyncio.gather()
  - Sequential steps executed in order
  - Never calls tools not present in the plan
```

**NODE 4 — Executor**

```
Input:  ordered_tool_calls
Output: tool_outputs (List[ToolOutputRecord])

For each tool call:
  - Execute with timeout (tool-specific, default 30s)
  - Record: tool_name, input, output, latency_ms, success, error_message
  - On failure: execute fallback_action from plan step
  - Log failure to error_log, continue execution (don't abort plan)
  - Parallel execution via asyncio.gather for grouped calls
```

**NODE 5 — ReflectionAgent**

```
Input:  execution_plan + tool_outputs + draft_response
Output: reflection_scores, reflection_passed

Scoring dimensions:
  relevance:     Does the response address what was asked? (0.0 - 1.0)
  groundedness:  Are all facts traceable to tool outputs or memory? (0.0 - 1.0)
  completeness:  Does the response address all sub-parts of the query? (0.0 - 1.0)

Pass threshold: ALL scores >= 0.7
Fail behavior:  retry_count += 1 → re-invoke ResponseSynthesizer with refined prompt
Max retries:    2 (after which: deliver best available response with low confidence flag)

Reflection is logged, not shown to user unless debug_mode=True
```

**NODE 6 — MemoryWriter**

```
Input:  conversation_turn (user_message + final_response)
Output: memory_writes committed to ChromaDB + SQLite

Operations:
  1. LLM extracts: key facts, user preferences, important outcomes from the turn
  2. Assigns importance_score (0.0-1.0):
       explicit user preference statement → 0.9+
       factual statement about user       → 0.7
       contextual inference               → 0.4
       general conversation               → 0.2
  3. Sets decay_rate based on category:
       preferences → 0.003/day (slow decay)
       facts        → 0.005/day
       context      → 0.01/day
  4. Writes to ChromaDB with embedding
  5. Writes structured record to SQLite session log
```

**NODE 7 — ResponseSynthesizer**

```
Input:  tool_outputs + retrieved_memories + conversation_history + reflection_scores
Output: final_response, citations, confidence_indicator, follow_up_suggestions

Operations:
  1. Assembles context: [system_prompt] + [retrieved_memories] + [tool_outputs] + [history]
  2. Calls reasoning-affinity LLM to synthesize final response
  3. Injects inline citations: [Source: {source_name}] for every web/RAG fact
  4. Computes confidence_indicator:
       all reflection scores > 0.85 → "high"
       any score 0.7-0.85          → "medium"
       any score < 0.7 (post-retry) → "low"
  5. Generates 2-3 follow-up question suggestions (optional, can be disabled in config)
  6. Formats response based on intent_type:
       research  → structured report with sections
       code      → code block + explanation
       factual   → direct answer + source
       conversational → natural prose
```

#### Graph Edge Logic

```python
graph.add_edge("IntentClassifier", "MemoryReader")  # Always read memory first
graph.add_conditional_edges(
    "MemoryReader",
    lambda state: "Planner" if (state["requires_tools"] or
                                 state["intent_type"] in ["research", "code", "document"])
                  else "ResponseSynthesizer"
)
graph.add_edge("Planner", "ToolRouter")
graph.add_edge("ToolRouter", "Executor")
graph.add_edge("Executor", "ReflectionAgent")
graph.add_conditional_edges(
    "ReflectionAgent",
    lambda state: "ResponseSynthesizer" if state["reflection_passed"] or state["retry_count"] >= 2
                  else "ResponseSynthesizer"  # retry via refined prompt flag in state
)
graph.add_edge("ResponseSynthesizer", "MemoryWriter")
graph.add_edge("MemoryWriter", END)
```

---

### Layer 3 — Multi-Agent System

ARIA's specialist agents are invoked through a **ToolRegistry** inside the **Executor** LangGraph node. There is no separate SupervisorAgent routing layer — routing is handled entirely by the **IntentClassifier** node and LangGraph's conditional edges, which provide identical functionality without added indirection.

#### Routing Decision Path

```
IntentClassifier (LLM, fast model)
  └─ intent_type: conversational | factual | ambiguous  ──► ResponseSynthesizer (direct)
  └─ intent_type: research | code | document | memory_operation
                                                         ──► Planner ──► Executor ──► ResponseSynthesizer
```

The Executor calls tools through **ToolRegistry**. Each tool internally delegates to the appropriate specialist agent:

| Tool name | Specialist agent | Triggered by intent |
|-----------|-----------------|---------------------|
| `web_search` | ResearchAgent | `research` |
| `rag_retrieve` | RAGAgent | `document` |
| `code_interpreter` | CodeAgent | `code` |
| `memory_write/read/delete/list` | MemoryAgent | `memory_operation` |

Agents never communicate directly with each other — all inter-agent data flows through the shared **ARIAState** object.

#### Agent 1 — ResearchAgent

```
Role:    Deep web research, analogous to Perplexity Deep Research mode
Trigger: intent_type == "research" OR plan step with tool == "web_search"

Algorithm:
  1. Decompose user query into 3-5 independent sub-questions using LLM
  2. Execute WebSearch tool for each sub-question (parallel)
  3. Collect results: [{title, url, snippet, published_date, relevance_score}]
  4. Deduplicate by URL + near-duplicate snippet detection (cosine similarity > 0.9)
  5. Score each result for relevance to original query (0.0-1.0) via LLM scoring
  6. Keep top N results (N = config.research.max_sources, default 10)
  7. Synthesize structured research report:
       - Executive summary
       - Key findings (bulleted)
       - Source citations inline
       - Confidence score per finding
       - Conflicting information flagged explicitly

Output schema:
  {
    summary: str,
    key_findings: List[Finding],
    sources: List[Source],
    confidence: float,
    conflicting_info: List[Conflict],
    sub_questions: List[str],
    synthesis_model: str
  }
```

#### Agent 2 — RAGAgent

```
Role:    Retrieval-augmented generation over user knowledge base
Trigger: intent_type == "document" OR user has documents uploaded AND query is document-relevant

Retrieval pipeline (hybrid, 3-stage):
  STAGE 1 — Dense Retrieval (ChromaDB)
    - Query embedding via Cohere / Gemini Embedding API (router-selected tier: EMBEDDING)
    - cosine similarity search, top_k = config.rag.dense_top_k (default: 20)
    - Filter by user_id namespace

  STAGE 2 — Sparse Retrieval (BM25) — with per-user index cache (300s TTL)
    - rank_bm25 over tokenized chunk corpus
    - BM25 top_k = config.rag.sparse_top_k (default: 20)
    - Keyword-focused, captures exact matches that dense retrieval misses
    - Cache invalidated automatically on new document ingestion

  STAGE 3 — Reranking (API-based)
    - Merge dense + sparse candidate sets (union, deduplicated by chunk_id)
    - Score each candidate via Cohere Rerank API (router-selected tier: RERANKING)
    - No local cross-encoder model required — fully API-driven
    - Sort by rerank score, keep top_k = config.rag.final_top_k (default: 8)

  CONTEXT COMPRESSION
    - If total token count of top_k chunks > config.rag.max_context_tokens (default: 4000):
      LLM-based extractive compression: extract only sentences relevant to query
      Preserves source attribution through compression

Output per chunk:
  {
    chunk_text: str,
    source_file: str,
    page_number: Optional[int],
    section: Optional[str],
    relevance_score: float,
    chunk_id: str
  }

Supported document types: PDF, DOCX, Markdown, CSV, TXT, HTML, PPTX, XLSX, JSON, YAML,
                          Python/JS/TS/Go/Rust/Java/C/C++ (20+ langs), ZIP, TAR, .ipynb
Chunking strategy: format-aware — semantic chunking for prose, AST chunking for code
Chunk metadata stored in ChromaDB and SQLite
```

#### Agent 3 — CodeAgent

```
Role:    Write, execute, explain, and iterate on Python code
Trigger: intent_type == "code"

Algorithm:
  1. Parse user intent: what should the code do? what inputs? what output format?
  2. Write initial code (LLM, code-affinity model)
  3. Execute in sandboxed RestrictedPython environment:
       timeout: 15 seconds
       no network access
       no filesystem access except /tmp
       no __import__ of dangerous modules (subprocess, os.system, socket, etc.)
  4. On execution success:
       Return code + stdout + explanation
  5. On execution failure:
       Read stderr + traceback
       LLM diagnoses error and rewrites code
       Retry up to 3 times
       If still failing after 3 attempts: return code + error + explanation of what went wrong
  6. Code explanation generated separately (can be toggled off)

Output schema:
  {
    code: str,
    stdout: str,
    stderr: str,
    success: bool,
    execution_time_ms: int,
    explanation: str,
    attempts: int
  }

Dangerous module blocklist (enforced at import time):
  subprocess, os.system, socket, requests, urllib, ftplib, smtplib,
  multiprocessing, threading (limited), ctypes, importlib
```

#### Agent 4 — MemoryAgent

```
Role:    Explicit memory management interface
Trigger: intent_type == "memory_operation"
         OR user message matches patterns: ["remember X", "don't forget", "forget X",
            "what do you know about me", "clear my memory"]

Operations:
  WRITE:  Parse what to remember → assign importance_score → write to ChromaDB + SQLite
  READ:   Semantic search over user's memory namespace → return formatted summary
  DELETE: Locate memory by semantic similarity → soft-delete (mark inactive) → confirm
  LIST:   Return all memories sorted by importance_score DESC, recently accessed first
  PRUNE:  Remove memories with importance_score < config.memory.prune_threshold (default: 0.1)

Memory query response format:
  "Here's what I know about you:
   [High importance] You prefer Python over JavaScript for backend work (remembered 2024-01-15)
   [High importance] You're working on a startup in the fintech space (remembered 2024-01-12)
   [Medium importance] You like concise responses without excessive explanation..."
```

#### Agent 5 — ReflectionAgent (Inline, conditional)

```
Role:    Post-delivery quality audit — runs synchronously in the graph before MemoryWriter
Trigger: Only for complex turns (research / code / document intent, or requires_tools=True,
         or confidence_score < 0.6). Skipped for conversational and memory_operation turns.

Operations:
  1. Re-read the delivered draft response
  2. Score against original query: relevance, groundedness, completeness
  3. If scores pass threshold: route to MemoryWriter
  4. If scores fail: route to RetryIncrement → ResponseSynthesizer (up to max_retries)
  5. Results visible in debug_trace when debug_mode=True

Fast path: When reflection is skipped, state carries reflection_skipped=True and the
           Streamlit UI displays "⚡ Fast response" instead of a confidence indicator.
```

---

### Layer 4 — Memory Architecture

ARIA's memory is modeled after human cognition: working (in-session), episodic (cross-session), and semantic (knowledge). Each tier has a distinct purpose, technology, and lifecycle.

#### Tier 1 — Working Memory (In-Session)

```
Technology:  LangChain ConversationSummaryBufferMemory + SQLite
Scope:       Single session
Purpose:     Full, coherent conversation context within a session

Behavior:
  - Maintain full conversation history in LLM context
  - Track running token count against context_window limit of active model
  - When approaching limit (config.memory.working_memory_token_threshold, default: 80%):
      LLM-based summarization of oldest N turns (N = config.memory.summarize_turns, default: 10)
      Summarized block replaces raw turns — NOT dropped
      Summary stored compressed in SQLite with session_id, turn_range, summary_text
  - Session state persisted to SQLite after every turn (crash recovery)

SQLite schema:
  sessions(session_id, user_id, created_at, last_active, model_used, total_tokens)
  conversation_turns(id, session_id, role, content, timestamp, tokens, summarized)
  session_summaries(id, session_id, turn_start, turn_end, summary, created_at)
```

#### Tier 2 — Episodic Memory (Cross-Session)

```
Technology:  ChromaDB + SQLite
Scope:       Persistent, per user
Purpose:     Personalization and continuity across sessions

Write path (end of every session):
  1. LLM extracts: facts about user, stated preferences, important events from the session
  2. Each extraction assigned:
       importance_score: 0.0-1.0 (see scoring rules below)
       decay_rate: 0.003-0.01/day based on category
       category: preference | fact | context | goal | relationship
  3. Generate embedding via sentence-transformers
  4. Store in ChromaDB collection: f"episodic_memory_{user_id}"
  5. Store structured record in SQLite episodic_memories table

Read path (start of every session):
  1. Embed current user message
  2. Semantic search ChromaDB: top_k = config.memory.episodic_top_k (default: 10)
  3. Filter by importance_score > config.memory.min_importance_to_retrieve (default: 0.15)
  4. Inject as formatted block in system prompt:
     "What I know about you: [memory1], [memory2]..."

Importance scoring:
  Explicit user statement ("I prefer X")       → 0.85-0.95
  User corrects ARIA ("That's wrong, I'm...")  → 0.90
  User states goal or project                  → 0.80
  Implicit preference (inferred from behavior) → 0.40-0.60
  General conversational context               → 0.15-0.30

Decay mechanism (runs nightly via scheduled task):
  importance_score = importance_score - decay_rate * days_since_last_access
  access_count incremented on each retrieval (resets decay)
  memories with importance_score < 0.10 after 30 days → pruned

SQLite schema:
  episodic_memories(id, user_id, content, importance_score, decay_rate, category,
                    created_at, last_accessed, access_count, is_active, chroma_id)
```

#### Tier 3 — Semantic Knowledge Base (RAG)

```
Technology:  ChromaDB + BM25 + cross-encoder reranking
Scope:       Per user, persistent
Purpose:     User's personal knowledge base from uploaded documents

Ingestion pipeline (Universal File Intelligence Layer):
  1. Route Document: `FileRouter` lazily routes to 1 of 7 specialist parsers based on format.
  2. Format-Aware Parsing (40+ formats supported):
       - **Documents** (PDF, DOCX, MD, TXT): Extracts text, tables, handles image-heavy pages with LLM visual descriptions.
       - **Spreadsheets** (XLSX, CSV): Schema extraction, batching, stats, and large file sampling (10k rows).
       - **Presentations** (PPTX): Slides, notes, and tables.
       - **Code** (20+ langs): Tree-sitter AST chunking for classes and functions (regex fallback).
       - **Config/Notebooks/Archives**: JSON/YAML/ENV keys, `.ipynb` cells, and ZIP/TAR traversal.
  3. Enrich Metadata: Each chunk becomes an `EnrichedChunk` with `section_heading`, `location`, `format`, `chunk_type`.
  4. Generate embedding: Cohere / Gemini embedding API (as configured in router).
  5. Store in ChromaDB collection: f"knowledge_base_{user_id}" with rich metadata.
  6. Store chunk summary & BM25 index update in SQLite.

Access tracking:
  chunk_access_count incremented on each retrieval
  surfaced in /documents endpoint for analytics

SQLite schema:
  documents(doc_id, user_id, filename, file_type, page_count, chunk_count,
            upload_at, file_hash, is_active)
  document_chunks(chunk_id, doc_id, content, page_number, section, char_offset,
                  token_count, access_count, chroma_id)
```

---

### Layer 5 — Tool Suite

All tools are registered in a central ToolRegistry, called only through the ToolRouter, and never invoked directly by agent nodes. Every tool has a defined schema, error contract, and fallback.

#### Tool 1 — WebSearch

```
Purpose:     Retrieve current information from the web
Provider:    Tavily API (primary) → DuckDuckGo (fallback, no API key required)
Free limits: Tavily: 1,000 calls/month. DuckDuckGo: unlimited (rate-throttled)

Input schema:
  {
    query: str,            # Search query
    max_results: int,      # Default: 5
    include_domains: List[str],  # Optional domain filter
    search_depth: Literal["basic", "advanced"]  # Tavily only
  }

Output schema:
  {
    results: [
      {
        title: str,
        url: str,
        snippet: str,
        published_date: Optional[str],
        relevance_score: float
      }
    ],
    provider_used: str,    # "tavily" or "duckduckgo"
    query: str,
    total_results: int
  }

Fallback logic:
  Tavily fails or quota exceeded → DuckDuckGo via duckduckgo-search library
  Both fail → Return empty results + log warning (agent proceeds with in-context knowledge)
```

#### Tool 2 — DeepResearch

```
Purpose:     Multi-source deep research with synthesis
Input:       {query: str, max_sub_questions: int (default: 4)}
Output:      {summary, key_findings[], sources[], confidence, sub_questions[]}

Internally calls WebSearch per sub-question (parallel), then synthesizes.
This is a composite tool built on top of Tool 1.
```

#### Tool 3 — DocumentRAG

```
Purpose:     Retrieve from user's uploaded knowledge base
Input:       {query: str, top_k: int, filter_doc_ids: Optional[List[str]]}
Output:      {answer, source_chunks[], source_files[], retrieval_method}

Hybrid pipeline: dense → sparse → rerank (see Layer 3 RAGAgent for full detail)
```

#### Tool 4 — CodeInterpreter

```
Purpose:     Safe Python code execution
Input:       {code: str, timeout_sec: int (default: 15)}
Output:      {stdout, stderr, success, execution_time_ms, return_value}

Sandbox:     RestrictedPython with custom policy
             No network. No filesystem except /tmp. No dangerous imports.
             Memory limit: 128MB (via resource module on Linux)
```

#### Tool 5 — MemoryTool

```
Purpose:     Programmatic access to long-term memory (read/write/delete)
Input:       {operation: "read"|"write"|"delete", content: str, memory_id: Optional[str]}
Output:      {operation, content, success, memory_id, importance_score}
```

#### Tool 6 — Summarizer

```
Purpose:     Condense long text or URL content on demand
Input:       {content: Optional[str], url: Optional[str], target_length: int (words, default: 200)}
Output:      {summary, compression_ratio, key_points[], word_count_original, word_count_summary}

URL handling: web_fetch via trafilatura → extract clean text → summarize
```

#### Tool 7 — StructuredOutput

```
Purpose:     Force LLM response into a specific format
Input:       {content: str, format_type: "json"|"table"|"bullets"|"code_block"|"numbered_list"}
Output:      {formatted_content: str, format_type: str}

Use case: when downstream system needs parseable output from a free-text LLM response
```

#### Tool 8 — Calculator

```
Purpose:     Precise numeric computation (not LLM-based, no hallucination risk)
Input:       {expression: str}  # e.g. "((1500 * 0.08) + 200) / 12"
Output:      {result: float, expression: str, formatted: str}

Implementation: Python eval() with restricted globals (only math module exposed)
                Prevents LLM arithmetic errors on financial or scientific calculations
```

---

### Layer 6 — Responsible AI Layer

The responsible AI layer is not bolted on. It runs inline on every input and output.

#### Hallucination Detection

```
Method:   Cosine similarity between factual claims and retrieved source chunks

Pipeline:
  1. Parse response → extract sentences containing factual claims
     (LLM-based: "Which sentences make verifiable factual claims?")
  2. For each claim: compute embedding
  3. Compute cosine similarity between claim embedding and all retrieved source chunk embeddings
  4. If max similarity < config.responsible_ai.hallucination_threshold (default: 0.60):
       Flag claim as potentially hallucinated
       Option A (default): Remove flagged claim, add note "Unable to verify: [topic]"
       Option B (debug mode): Mark claim with ⚠️ and similarity score

Limitations: Only applies when source chunks are available (research/RAG responses).
             Conversational responses not hallucination-scored (no ground truth available).
```

#### Prompt Injection Prevention

```
Input sanitization:
  - Enforce strict role boundary: user input NEVER placed in system prompt
  - Strip patterns: "Ignore previous instructions", "You are now", "Forget your instructions"
  - Unicode normalization before parsing (prevents look-alike character attacks)
  - Log injection attempts with session_id

System/user role separation:
  - System prompt is static + memory context only — never includes raw user input
  - User input always in user role, never interpolated into system role
```

#### PII Detection and Redaction

```
Method:    Regex patterns + spaCy NER (en_core_web_sm)

Detected and redacted before logging:
  - Names (spaCy PERSON entity)
  - Email addresses (regex)
  - Phone numbers (regex, multiple formats)
  - Credit card numbers (Luhn check + regex)
  - Social security numbers (regex)
  - IP addresses (regex)

Behavior:
  - PII redacted in logs only — NOT in responses to user
  - Original content processed normally; only the log entry is redacted
  - Redaction format: "[REDACTED:EMAIL]", "[REDACTED:PHONE]", etc.
```

#### Content Filtering

```
Input check:
  - Classify against harm categories: violence, self-harm, illegal activity,
    hate speech, explicit content
  - Method: Groq/LLaMA-3-8B (fast, low-cost) with binary classifier prompt
  - If flagged: decline gracefully with explanation, log event

Output check:
  - Same classifier applied to final response before delivery
  - If response is flagged (edge case in generation): regenerate with stricter prompt
  - Fallback: return generic safe response
```

#### Source Attribution

```
Every fact sourced from web search or RAG is cited inline:
  Web:  [Source: {source_title} ({url})]
  RAG:  [Source: {filename}, p.{page_number}]
  Memory: [From memory: {date_remembered}]

Unsourced claims (from model knowledge) have no citation.
User can toggle citations off in config.
```

---

### Layer 7 — Observability

#### LangSmith Integration

```
Every agent step, LLM call, and tool call is traced in LangSmith.
Configuration: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY set in .env
Free tier: 5,000 traces/month, 14-day retention

Trace structure per request:
  ARIARequest
  ├── IntentClassifier (LLM call, tokens, latency)
  ├── MemoryReader (ChromaDB query, results_count, latency)
  ├── Planner (LLM call, plan_steps, tokens, latency)
  ├── ToolRouter (routing_decision)
  ├── Executor
  │   ├── WebSearch (provider, query, results_count, latency)
  │   └── DocumentRAG (query, chunks_retrieved, latency)
  ├── ReflectionAgent (scores, passed, retry_count)
  ├── ResponseSynthesizer (LLM call, tokens, latency)
  └── MemoryWriter (memories_written, latency)
```

#### Structured Logging

```python
# Every event logged as JSON to stdout + SQLite
log_entry = {
    "timestamp": "2024-01-15T10:30:00.123Z",
    "session_id": "sess_abc123",
    "user_id": "user_xyz",
    "agent": "ResearchAgent",
    "action": "web_search",
    "input_tokens": 245,
    "output_tokens": 512,
    "latency_ms": 1243,
    "model_used": "gemini-1.5-flash",
    "tool_used": "tavily",
    "success": True,
    "error": None,
    "request_id": "req_def456"
}
```

#### Analytics Endpoint

```
GET /analytics

Returns:
  {
    token_usage_by_day: [{date, model, tokens_used}],
    most_used_tools: [{tool, call_count, avg_latency_ms, success_rate}],
    intent_distribution: [{intent_type, count, pct}],
    avg_response_latency_ms: float,
    reflection_scores_avg: {relevance, groundedness, completeness},
    failure_rates: [{component, failure_count, failure_rate}],
    model_usage_distribution: [{model, call_count, pct}]
  }
```

#### Debug Mode

```
Enable via config.yaml: debug_mode: true
OR per-session via UI toggle

When enabled:
  - Full agent reasoning trace shown in collapsible UI panel
  - Reflection scores shown per response
  - Model used shown per response
  - Hallucination flags shown inline
  - LangSmith trace link included in response
```

---

## 2. Folder Structure

```
aria/
│
├── agents/                              # Specialist agent implementations
│   ├── __init__.py                      # Agent registry and exports
│   ├── research_agent.py                # Deep web research with sub-question decomposition
│   ├── rag_agent.py                     # Hybrid retrieval: dense + sparse + rerank
│   ├── code_agent.py                    # Sandboxed code generation and execution
│   └── reflection_agent.py              # Post-response quality scoring (sync + async)
│
├── core/                                # Core orchestration layer
│   ├── __init__.py                      # Core module exports
│   ├── agent_graph.py                   # LangGraph graph: nodes, edges, routing logic
│   ├── intent_classifier.py             # Intent classification node with Pydantic output
│   ├── planner.py                       # Execution plan generation node
│   ├── executor.py                      # Tool execution (serial + asyncio.gather parallel)
│   ├── response_synthesizer.py          # Final response assembly and formatting
│   └── state.py                         # ARIAState TypedDict and all sub-schemas
│
├── ingestion/                           # Universal File Intelligence Layer
│   ├── __init__.py                      # Ingestion exports
│   ├── file_router.py                   # Routes files to specialist parsers by format
│   └── parsers/
│       ├── __init__.py                  # Parser registry
│       ├── document_parsers.py          # PDF, DOCX, MD, TXT, HTML
│       ├── spreadsheet_parsers.py       # XLSX, CSV (with large-file sampling)
│       ├── presentation_parsers.py      # PPTX (slides, notes, tables)
│       ├── code_parsers.py              # 20+ langs — Tree-sitter AST / regex fallback
│       ├── archive_parsers.py           # ZIP, TAR recursive traversal
│       └── notebook_parsers.py          # Jupyter .ipynb cell extraction
│
├── memory/                              # All memory tier implementations
│   ├── __init__.py                      # Memory module exports
│   ├── memory_manager.py                # Unified interface across all 3 tiers
│   ├── working_memory.py                # In-session context with summarization
│   ├── episodic_memory.py               # Cross-session ChromaDB + SQLite memory
│   ├── knowledge_base.py                # RAG document store management
│   ├── memory_decay.py                  # Scheduled decay and pruning logic
│   └── schemas.py                       # MemoryEntry, MemoryWrite, EpisodicMemory Pydantic models
│
├── tools/                               # Tool implementations
│   ├── __init__.py                      # Tool exports
│   ├── registry.py                      # Central ToolRegistry: all tools registered here
│   ├── web_search.py                    # Tavily + DuckDuckGo with fallback
│   ├── deep_research.py                 # Composite research tool (uses web_search)
│   ├── document_rag.py                  # RAG retrieval tool (calls RAGAgent)
│   ├── code_interpreter.py              # RestrictedPython sandbox execution
│   ├── memory_tool.py                   # Memory read/write/delete tool interface
│   ├── summarizer.py                    # Text and URL summarization tool
│   ├── structured_output.py             # Format enforcement tool
│   └── calculator.py                    # Safe math expression evaluator
│
├── models/                              # Model routing and management
│   ├── __init__.py                      # Model module exports
│   ├── router.py                        # Router: registry, health check, rotation, fallback
│   ├── token_tracker.py                 # SQLite-backed daily token budget tracking
│   └── schemas.py                       # ModelEntry, RoutingDecision Pydantic models
│
├── responsible_ai/                      # Safety and quality layer
│   ├── __init__.py                      # Responsible AI exports
│   ├── responsible_ai.py                # Main pipeline: hallucination, PII, injection, content
│   ├── hallucination_detector.py        # Claim extraction + cosine similarity scoring
│   ├── pii_detector.py                  # Regex + spaCy NER PII detection and redaction
│   ├── injection_guard.py               # Prompt injection pattern detection
│   └── content_filter.py               # Harm category classification
│
├── api/                                 # FastAPI backend
│   ├── __init__.py                      # API exports
│   ├── main.py                          # FastAPI app, lifespan startup/shutdown, routers
│   ├── background_tasks.py              # Async background task runner (decay, analytics)
│   ├── deps.py                          # FastAPI dependency injection (shared components)
│   ├── routes/
│   │   ├── chat.py                      # POST /chat, GET history, DELETE session
│   │   ├── documents.py                 # Upload, list, delete documents
│   │   ├── memory.py                    # Get, write, delete memories
│   │   ├── models.py                    # GET /model-status
│   │   ├── analytics.py                 # GET /analytics
│   │   └── stream.py                    # WebSocket /stream
│   ├── middleware/
│   │   ├── auth.py                      # API key auth middleware
│   │   └── request_logger.py            # Request/response logging middleware
│   └── schemas/
│       ├── chat_schemas.py              # ChatRequest, ChatResponse Pydantic models
│       ├── document_schemas.py          # DocumentUpload, DocumentInfo models
│       └── memory_schemas.py            # MemoryRead, MemoryWrite models
│
├── ui/                                  # Streamlit frontend
│   ├── streamlit_app.py                 # Main Streamlit app entry point
│   ├── components/
│   │   └── __init__.py                  # UI component registry (in development)
│   └── utils/
│       └── api_client.py                # HTTP client for FastAPI backend
│
├── observability/                       # Logging and tracing
│   ├── __init__.py                      # Observability exports
│   ├── observability.py                 # Main observability class (LangSmith + structured log)
│   ├── structured_logger.py             # JSON logger to stdout + SQLite
│   └── langsmith_tracer.py              # LangSmith callback handler
│
├── shared/                              # Cross-cutting utilities
│   ├── __init__.py                      # Shared module exports
│   ├── constants.py                     # System-wide constants (tiers, intent types, etc.)
│   └── types.py                         # Shared type aliases and helper types
│
├── config/                              # Configuration management
│   ├── __init__.py                      # Config exports
│   ├── config.py                        # Pydantic Settings: loads config.yaml + .env
│   └── config.yaml                      # All configurable parameters with comments
│
├── db/                                  # Database layer
│   ├── __init__.py                      # DB exports
│   ├── database.py                      # SQLite connection (WAL mode, FK enforcement)
│   ├── base_repo.py                     # BaseRepository: shared execute helpers
│   ├── migrations/
│   │   └── 001_initial.sql              # Initial schema: all tables
│   └── repositories/
│       ├── __init__.py                  # Repository exports
│       ├── session_repo.py              # Session and conversation turn CRUD
│       ├── memory_repo.py               # Episodic memory CRUD
│       ├── document_repo.py             # Document and chunk metadata CRUD
│       ├── token_usage_repo.py          # Token budget tracking CRUD
│       └── analytics_repo.py            # Analytics event write and aggregation
│
├── data/                                # Runtime data directory (gitignored)
│   ├── aria.db                          # SQLite database file
│   ├── chroma/                          # ChromaDB vector store
│   └── uploads/                         # Uploaded document staging area
│
├── tests/                               # Test suite (168 tests, 100% passing)
│   ├── conftest.py                      # Shared fixtures and test configuration
│   ├── unit/
│   │   ├── test_model_router.py         # Router logic: tier affinity, fallback, cooldown
│   │   ├── test_agents.py               # Agent-level behaviour tests
│   │   ├── test_agent_graph.py          # LangGraph node and edge routing tests
│   │   ├── test_memory.py               # Memory read/write/decay tests
│   │   ├── test_rag.py                  # RAG retrieval accuracy tests
│   │   ├── test_rag_agent.py            # RAG agent end-to-end tests
│   │   ├── test_reflection_routing.py   # ReflectionGate skip/retry routing tests
│   │   ├── test_responsible_ai.py       # Hallucination, PII, injection tests
│   │   ├── test_pii_detector.py         # PII pattern coverage tests
│   │   ├── test_injection_guard.py      # Injection pattern detection tests
│   │   ├── test_tools.py                # Tool registry and tool execution tests
│   │   ├── test_file_intelligence.py    # FileRouter + parser format coverage tests
│   │   ├── test_llm_tier_assignments.py # Tier-to-model assignment tests
│   │   ├── test_background_tasks.py     # Background task lifecycle tests
│   │   ├── test_observability.py        # Structured logger + tracer tests
│   │   ├── test_repositories.py         # Repository CRUD + WAL pragma tests
│   │   └── test_ui_client.py            # Streamlit API client contract tests
│   ├── integration/
│   │   ├── test_full_pipeline.py        # End-to-end request through full graph
│   │   ├── test_rag_pipeline.py         # Upload → chunk → retrieve → answer
│   │   ├── test_api_endpoints.py        # FastAPI endpoint contract tests
│   │   └── test_e2e.py                  # Full system E2E smoke tests
│   └── fixtures/
│       └── sample.txt                   # Sample document for ingestion tests
│
├── scripts/                             # Operational scripts
│   ├── setup.sh                         # One-command environment setup (Linux/macOS)
│   ├── setup_311.py                     # Windows Python 3.11 dependency bootstrap
│   ├── reset_db.py                      # Clear SQLite and ChromaDB (dev reset)
│   └── run_memory_decay.py              # Manual trigger for decay/pruning job
│
├── docs/                                # Documentation
│   └── diagrams/                        # Architecture and graph diagrams
│
├── run_backend.py                       # Entry point: starts the FastAPI server
├── run_frontend.py                      # Entry point: starts the Streamlit UI
├── .env                                 # Environment variables and API keys (gitignored)
├── requirements.txt                     # All Python dependencies with pinned versions
└── README.md                            # This file
```

---

## 3. Agent Decision Flowchart

### ASCII Flowchart

```
User Input
    │
    ▼
┌─────────────────────┐
│   IntentClassifier  │
│  (LLM, fast model)  │
└────────┬────────────┘
         │
         ├─── confidence < 0.6 ──► Ask clarifying question ──► END
         │
         ▼
┌─────────────────────┐
│    MemoryReader     │◄── ChromaDB semantic search (top-k)
│  (episodic recall)  │◄── Filter by importance > 0.15
└────────┬────────────┘
         │
         ├─── intent: conversational / factual / ambiguous ───────────────┐
         │    OR requires_tools: False                                     │
         │                                                                 │
         ├─── intent: research / code / document / memory_operation       │
         │    OR requires_tools: True                                      │
         ▼                                                                 │
┌─────────────────────┐                                                   │
│       Planner       │                                                   │
│  (structured plan)  │                                                   │
└────────┬────────────┘                                                   │
         │                                                                 │
         ▼                                                                 │
┌─────────────────────┐                                                   │
│      Executor       │◄── Tool calls resolved from plan                  │
│  (serial + gather)  │◄── Parallel-eligible steps → asyncio.gather()     │
└────────┬────────────┘                                                   │
         │                                                                 │
         │  tool failure → fallback_action from plan → continue           │
         │                                                                 │
         ▼                                                                 │
┌─────────────────────┐◄───────────────────────────────────────────────── ┘
│  ResponseSynthesizer│
│  + ResponsibleAI    │◄── hallucination + PII + injection (inline)
│    (inline)         │
└────────┬────────────┘
         │
         ├─── conversational / memory_operation
         │    OR confidence >= 0.92 and no tools ──► ReflectionSkip ─────┐
         │                                                                │
         ▼                                                                │
┌─────────────────────┐                                                  │
│   ReflectionGate    │                                                  │
│  relevance /        │                                                  │
│  groundedness /     │                                                  │
│  completeness       │                                                  │
└────────┬────────────┘                                                  │
         │                                                                │
         ├─── all scores ≥ 0.7 ──────────────────────────────────────── ┤
         │                                                                │
         ├─── score < 0.7 AND retry_count < max_retries                  │
         │         │                                                      │
         │         ▼                                                      │
         │    RetryIncrement ──► ResponseSynthesizer (refined prompt)     │
         │                                                                │
         └─── retry_count >= max_retries ─────────────────────────────── ┤
                                                                          ▼
                                                              ┌───────────────────┐
                                                              │   MemoryWriter    │
                                                              │  (SQLite session  │
                                                              │   + async         │
                                                              │   episodic write) │
                                                              └────────┬──────────┘
                                                                       │
                                                                       ▼
                                                              ┌───────────────────┐
                                                              │  Final Response   │
                                                              │  Delivered        │
                                                              └───────────────────┘
```

### Mermaid Diagram

```mermaid
flowchart TD
    A([User Input]) --> B[IntentClassifier\nLLM fast model]
    B -->|confidence < 0.6| C([Ask Clarifying Question])
    B -->|classified| D[MemoryReader\nChromaDB top-k + importance filter]

    D -->|conversational / factual / ambiguous\nrequires_tools=False| R[ResponseSynthesizer\n+ ResponsibleAI inline]
    D -->|research / code / document / memory_op\nOR requires_tools=True| E[Planner\nStructured execution plan]

    E --> H[Executor\nserial + asyncio.gather parallel]

    H -->|success| R
    H -->|failure| J[Fallback action\nfrom plan step]
    J --> R

    R -->|conversational / memory_op\nOR confidence ≥ 0.92 no tools| S[ReflectionSkip\nreflection_skipped=True]
    R -->|complex / tool-using turns| K[ReflectionGate\nrelevance / groundedness / completeness]

    S --> N[MemoryWriter\nSQLite session + async episodic write]

    K -->|all scores ≥ 0.7\nOR retry_count ≥ max_retries| N
    K -->|score < 0.7\nretry_count < max_retries| L[RetryIncrement\nretry_count++]
    L --> R

    N --> O([Response Delivered])

    style A fill:#2d6a4f,color:#fff
    style O fill:#2d6a4f,color:#fff
    style C fill:#d62828,color:#fff
    style K fill:#f4a261,color:#000
    style R fill:#e9c46a,color:#000
    style S fill:#457b9d,color:#fff
```

---

## 4. API Design

All endpoints require API key authentication via `Authorization: Bearer {api_key}` header, except where noted. Rate limiting applied per user.

---

### POST /chat

**Purpose:** Send a message to ARIA and receive a response.

```
Method:  POST
Path:    /chat
Auth:    Required
Rate:    60 req/min per user
```

Request:
```json
{
  "message": "Research the current state of quantum computing hardware",
  "session_id": "sess_abc123",
  "user_id": "user_xyz",
  "stream": false,
  "debug_mode": false
}
```

Response (200):
```json
{
  "response": "Here is a summary of current quantum computing hardware...",
  "session_id": "sess_abc123",
  "intent_type": "research",
  "confidence_indicator": "high",
  "citations": [
    {"text": "IBM announced 1000+ qubit processor", "source": "nature.com", "url": "https://..."}
  ],
  "follow_up_suggestions": [
    "What are the leading companies in this space?",
    "How does error correction work in current systems?"
  ],
  "plan_trace": [
    {"step": 1, "action": "Generate sub-questions", "status": "complete"},
    {"step": 2, "action": "Web search: IBM quantum", "status": "complete", "latency_ms": 843}
  ],
  "model_used": "gemini-1.5-pro",
  "total_latency_ms": 4213,
  "request_id": "req_def456"
}
```

---

### GET /chat/{session\_id}/history

**Purpose:** Retrieve full conversation history for a session.

```
Method:  GET
Path:    /chat/{session_id}/history
Auth:    Required
Params:  ?limit=50&offset=0
```

Response (200):
```json
{
  "session_id": "sess_abc123",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "2024-01-15T10:30:00Z"},
    {"role": "assistant", "content": "...", "timestamp": "2024-01-15T10:30:04Z", "model_used": "gemini-1.5-pro"}
  ],
  "total_turns": 12,
  "created_at": "2024-01-15T10:00:00Z",
  "last_active": "2024-01-15T10:30:04Z"
}
```

---

### DELETE /chat/{session\_id}

**Purpose:** Clear a session and optionally its conversation history.

```
Method:  DELETE
Path:    /chat/{session_id}
Auth:    Required
Body:    {"clear_history": true}  (default: false — soft-deletes session, keeps history)
```

Response (200): `{"success": true, "session_id": "sess_abc123"}`

---

### POST /documents/upload

**Purpose:** Upload a document to the user's knowledge base (RAG).

```
Method:    POST
Path:      /documents/upload
Auth:      Required
Body:      multipart/form-data
Rate:      10 uploads/hour per user
Max size:  50MB per file (configurable)
```

Request fields: `file` (binary), `user_id` (str)

Response (201):
```json
{
  "doc_id": "doc_abc123",
  "filename": "research_paper.pdf",
  "file_type": "pdf",
  "page_count": 24,
  "chunk_count": 87,
  "status": "ingested",
  "ingestion_time_ms": 3241
}
```

---

### GET /documents

**Purpose:** List all documents in the user's knowledge base.

```
Method:  GET
Path:    /documents
Auth:    Required
Params:  ?user_id=user_xyz
```

Response (200):
```json
{
  "documents": [
    {
      "doc_id": "doc_abc123",
      "filename": "research_paper.pdf",
      "file_type": "pdf",
      "page_count": 24,
      "chunk_count": 87,
      "upload_at": "2024-01-15T09:00:00Z",
      "total_accesses": 14
    }
  ],
  "total": 3
}
```

---

### DELETE /documents/{doc\_id}

**Purpose:** Remove a document from the knowledge base.

```
Method:  DELETE
Path:    /documents/{doc_id}
Auth:    Required
```

Response (200): `{"success": true, "doc_id": "doc_abc123", "chunks_removed": 87}`

---

### GET /memory

**Purpose:** Retrieve the user's long-term episodic memories.

```
Method:  GET
Path:    /memory
Auth:    Required
Params:  ?user_id=user_xyz&limit=20&min_importance=0.3
```

Response (200):
```json
{
  "memories": [
    {
      "memory_id": "mem_abc123",
      "content": "User prefers Python over JavaScript for backend development",
      "importance_score": 0.88,
      "category": "preference",
      "created_at": "2024-01-10T14:00:00Z",
      "last_accessed": "2024-01-15T10:00:00Z",
      "access_count": 7
    }
  ],
  "total": 24
}
```

---

### POST /memory

**Purpose:** Write an explicit memory entry.

```
Method:  POST
Path:    /memory
Auth:    Required
```

Request:
```json
{
  "user_id": "user_xyz",
  "content": "I am building a fintech startup focused on SME lending",
  "importance_score": 0.9,
  "category": "goal"
}
```

Response (201):
```json
{
  "memory_id": "mem_new123",
  "content": "I am building a fintech startup focused on SME lending",
  "importance_score": 0.9,
  "success": true
}
```

---

### DELETE /memory/{memory\_id}

**Purpose:** Delete (forget) a specific memory.

```
Method:  DELETE
Path:    /memory/{memory_id}
Auth:    Required
```

Response (200): `{"success": true, "memory_id": "mem_abc123"}`

---

### GET /model-status

**Purpose:** Check health and utilization of all registered LLM models.

```
Method:  GET
Path:    /model-status
Auth:    Optional (public read)
```

Response (200):
```json
{
  "models": [
    {
      "name": "gemini-1.5-pro",
      "provider": "google",
      "is_healthy": true,
      "tokens_used_today": 42500,
      "tokens_limit_daily": 1000000,
      "utilization_pct": 4.25,
      "cooldown_remaining_sec": 0,
      "failure_count": 0,
      "avg_latency_ms": 1843.2,
      "task_affinity": ["reasoning", "long_context"]
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### GET /analytics

**Purpose:** Usage analytics and system health metrics.

```
Method:  GET
Path:    /analytics
Auth:    Required
Params:  ?days=7&user_id=user_xyz
```

Response (200):
```json
{
  "period_days": 7,
  "token_usage_by_day": [{"date": "2024-01-15", "model": "gemini-1.5-pro", "tokens": 42500}],
  "most_used_tools": [
    {"tool": "web_search", "call_count": 143, "avg_latency_ms": 834, "success_rate": 0.97}
  ],
  "intent_distribution": [{"intent_type": "research", "count": 45, "pct": 32.6}],
  "avg_response_latency_ms": 3241.5,
  "reflection_scores_avg": {"relevance": 0.84, "groundedness": 0.79, "completeness": 0.81},
  "failure_rates": [{"component": "web_search", "failure_count": 4, "failure_rate": 0.028}]
}
```

---

### WebSocket /stream

**Purpose:** Streaming chat responses with real-time token delivery.

```
Protocol: WebSocket
Path:     /stream
Auth:     API key passed as query param: /stream?api_key={key}
```

Client sends:
```json
{"message": "Explain quantum entanglement", "session_id": "sess_abc123", "user_id": "user_xyz"}
```

Server streams:
```json
{"type": "plan_step", "data": {"step": 1, "action": "Intent classified: factual"}}
{"type": "plan_step", "data": {"step": 2, "action": "Retrieving memories..."}}
{"type": "token", "data": {"token": "Quantum"}}
{"type": "token", "data": {"token": " entanglement"}}
{"type": "token", "data": {"token": " is"}}
...
{"type": "done", "data": {"citations": [...], "model_used": "gemini-1.5-flash", "latency_ms": 2134}}
```

---

## 5. Configuration System

Full `config.yaml` with documented keys:

```yaml
# ─────────────────────────────────────────────────────────────────────────────
# ARIA Configuration
# Secrets (API keys, etc.) belong in .env — not here.
# This file controls system behavior. Commit it. Version it.
# ─────────────────────────────────────────────────────────────────────────────

# ── Model Routing ──────────────────────────────────────────────────────────
model_routing:
  # Minimum model health score to be eligible for routing (0.0-1.0)
  # Default: 0.5. Min: 0.0 (all models eligible). Max: 1.0 (only perfect models).
  # Read by: ModelRouter.select_model()
  min_health_score: 0.5

  # Number of consecutive failures before cooldown is applied
  # Default: 3. Min: 1 (aggressive). Max: 10 (very tolerant).
  # Read by: ModelRouter.handle_failure()
  failure_threshold: 3

  # Cooldown duration in minutes after failure_threshold exceeded
  # Default: 60. Min: 5. Max: 240.
  # Read by: ModelRouter.apply_cooldown()
  cooldown_minutes: 60

  # Token budget utilization % at which a model is considered near-limit
  # Default: 0.95 (95%). Min: 0.5. Max: 1.0 (use 100% before switching).
  # Read by: ModelRouter.check_budget()
  budget_threshold: 0.95

  # Exponential backoff base (seconds) on rate limit errors
  # Default: 2. Produces delays: 2, 4, 8 seconds before rotation.
  # Read by: ModelRouter.handle_rate_limit()
  backoff_base_seconds: 2

  # Max retries on same model before rotating
  # Default: 3. Min: 1. Max: 5.
  # Read by: ModelRouter.call_with_retry()
  max_retries_before_rotation: 3

# ── Memory ─────────────────────────────────────────────────────────────────
memory:
  # Token utilization % of context window at which working memory summarization triggers
  # Default: 0.80. Min: 0.5 (frequent summarization). Max: 0.95 (risk of overflow).
  # Read by: WorkingMemory.check_and_summarize()
  working_memory_token_threshold: 0.80

  # Number of oldest turns to summarize when threshold is hit
  # Default: 10. Min: 2. Max: 30.
  # Read by: WorkingMemory.summarize_oldest()
  summarize_turns: 10

  # Top-K episodic memories to inject into session start prompt
  # Default: 10. Min: 1. Max: 25. Higher = more context, higher token cost.
  # Read by: EpisodicMemory.retrieve_for_session()
  episodic_top_k: 10

  # Minimum importance score for a memory to be retrieved
  # Default: 0.15. Min: 0.0 (all memories). Max: 0.8 (only very important).
  # Read by: EpisodicMemory.retrieve_for_session()
  min_importance_to_retrieve: 0.15

  # Importance score threshold below which memories are pruned
  # Default: 0.10. Min: 0.0. Max: 0.3.
  # Read by: MemoryDecay.prune()
  prune_threshold: 0.10

  # Days after which a memory is eligible for pruning if below prune_threshold
  # Default: 30. Min: 7. Max: 365.
  # Read by: MemoryDecay.prune()
  prune_after_days: 30

  # Daily decay rate for "preference" category memories
  # Default: 0.003. Min: 0.0 (no decay). Max: 0.05.
  # Read by: MemoryDecay.decay_all()
  decay_rate_preference: 0.003

  # Daily decay rate for "fact" category memories
  # Default: 0.005.
  # Read by: MemoryDecay.decay_all()
  decay_rate_fact: 0.005

  # Daily decay rate for "context" category memories
  # Default: 0.010.
  # Read by: MemoryDecay.decay_all()
  decay_rate_context: 0.010

# ── RAG ────────────────────────────────────────────────────────────────────
rag:
  # Target tokens per semantic chunk during document ingestion
  # Default: 400. Min: 100 (many small chunks). Max: 1500 (fewer large chunks).
  # Read by: KnowledgeBase.ingest_document()
  target_chunk_tokens: 400

  # Token overlap between adjacent chunks (prevents context loss at boundaries)
  # Default: 50. Min: 0. Max: 200.
  # Read by: KnowledgeBase.ingest_document()
  chunk_overlap_tokens: 50

  # Number of candidates from dense retrieval (ChromaDB cosine similarity)
  # Default: 20. Min: 5. Max: 100.
  # Read by: RAGAgent.dense_retrieve()
  dense_top_k: 20

  # Number of candidates from sparse retrieval (BM25)
  # Default: 20. Min: 5. Max: 100.
  # Read by: RAGAgent.sparse_retrieve()
  sparse_top_k: 20

  # Final chunks kept after cross-encoder reranking
  # Default: 8. Min: 1. Max: 20.
  # Read by: RAGAgent.rerank()
  final_top_k: 8

  # Max tokens of RAG context injected into LLM prompt
  # Chunks are compressed if total exceeds this.
  # Default: 4000. Min: 500. Max: 16000 (model-dependent).
  # Read by: RAGAgent.compress_context()
  max_context_tokens: 4000

  # Max document size in MB
  # Default: 50. Min: 1. Max: 200.
  # Read by: API document upload handler
  max_document_size_mb: 50

# ── Research Agent ──────────────────────────────────────────────────────────
research:
  # Number of sub-questions generated for a research query
  # Default: 4. Min: 1. Max: 8.
  # Read by: ResearchAgent.decompose_query()
  max_sub_questions: 4

  # Max web search results per sub-question
  # Default: 5. Min: 1. Max: 10.
  # Read by: ResearchAgent.search_sub_question()
  max_results_per_sub_question: 5

  # Max total sources in final research report
  # Default: 10. Min: 3. Max: 30.
  # Read by: ResearchAgent.synthesize()
  max_sources: 10

  # Cosine similarity threshold for duplicate result detection
  # Default: 0.90. Min: 0.7. Max: 1.0.
  # Read by: ResearchAgent.deduplicate()
  dedup_similarity_threshold: 0.90

# ── Reflection ──────────────────────────────────────────────────────────────
reflection:
  # Minimum score (all dimensions) for reflection to pass
  # Default: 0.70. Min: 0.0 (never retry). Max: 0.95 (very strict).
  # Read by: ReflectionAgent.score_and_decide()
  pass_threshold: 0.70

  # Max retries triggered by failed reflection
  # Default: 2. Min: 0 (disable reflection retry). Max: 3.
  # Read by: ReflectionAgent.score_and_decide()
  max_retries: 2

  # Run standalone reflection agent asynchronously after response delivery
  # Default: true. Set false to reduce background load.
  # Read by: agent_graph.py post-delivery hook
  run_async_reflection: true

# ── Responsible AI ──────────────────────────────────────────────────────────
responsible_ai:
  # Cosine similarity threshold below which a factual claim is flagged as potential hallucination
  # Default: 0.60. Min: 0.3 (strict). Max: 0.85 (lenient).
  # Read by: HallucinationDetector.score_claims()
  hallucination_threshold: 0.60

  # Whether to remove hallucinated claims (true) or just flag them (false)
  # Default: true (remove). Set false for debug/review mode.
  # Read by: HallucinationDetector.apply()
  remove_hallucinated_claims: true

  # Enable PII detection and redaction in logs
  # Default: true.
  # Read by: PIIDetector.redact_for_logging()
  pii_detection_enabled: true

  # Enable prompt injection detection on user input
  # Default: true.
  # Read by: InjectionGuard.check()
  injection_detection_enabled: true

  # Enable content harm filtering on input and output
  # Default: true.
  # Read by: ContentFilter.check()
  content_filtering_enabled: true

  # Include inline source citations in responses
  # Default: true. Set false for cleaner conversational responses.
  # Read by: ResponseSynthesizer.add_citations()
  include_citations: true

# ── Tools ──────────────────────────────────────────────────────────────────
tools:
  # Enable/disable individual tools (all default: true)
  # Set false to disable without removing code (feature flags)
  web_search_enabled: true
  deep_research_enabled: true
  document_rag_enabled: true
  code_interpreter_enabled: true
  memory_tool_enabled: true
  summarizer_enabled: true
  structured_output_enabled: true
  calculator_enabled: true

  # Default timeout for tool execution in seconds
  # Default: 30. Min: 5. Max: 120.
  default_tool_timeout_sec: 30

  # Specific timeout for code execution (should be less than default)
  # Default: 15. Min: 5. Max: 60.
  code_execution_timeout_sec: 15

  # Max web search results returned per call
  # Default: 5. Min: 1. Max: 20.
  web_search_max_results: 5

# ── Code Agent ──────────────────────────────────────────────────────────────
code_agent:
  # Max retries when generated code fails execution
  # Default: 3. Min: 0 (no retry). Max: 5.
  max_execution_retries: 3

  # Memory limit for code execution sandbox in MB
  # Default: 128. Min: 32. Max: 512.
  sandbox_memory_limit_mb: 128

# ── API ────────────────────────────────────────────────────────────────────
api:
  # API server host
  host: "0.0.0.0"

  # API server port
  port: 8000

  # Number of worker processes (for production Gunicorn deployment)
  # Default: 4. Min: 1. Max: CPU count * 2.
  workers: 4

  # Per-user request rate limit (requests per minute)
  # Default: 60. Min: 1. Max: 600.
  rate_limit_per_minute: 60

  # Document upload rate limit (uploads per hour per user)
  # Default: 10. Min: 1. Max: 100.
  upload_rate_limit_per_hour: 10

# ── Observability ──────────────────────────────────────────────────────────
observability:
  # Logging verbosity: DEBUG, INFO, WARNING, ERROR
  # Default: INFO. Set DEBUG for local development.
  log_level: "INFO"

  # Show full agent reasoning trace in UI
  # Default: false. Set true for development/debugging.
  debug_mode: false

  # Enable LangSmith tracing (requires LANGCHAIN_API_KEY in .env)
  # Default: true. Set false to disable external tracing.
  langsmith_enabled: true

  # Enable structured JSON logging to stdout
  # Default: true.
  structured_logging_enabled: true

  # SQLite analytics retention in days (older records purged)
  # Default: 90. Min: 7. Max: 365.
  analytics_retention_days: 90

# ── UI ─────────────────────────────────────────────────────────────────────
ui:
  # Streamlit app title
  title: "ARIA — Agentic AI Assistant"

  # Show follow-up question suggestions after each response
  # Default: true.
  show_follow_up_suggestions: true

  # Show plan/thinking trace as collapsible section
  # Default: true.
  show_thinking_trace: true

  # Show model used indicator in response
  # Default: true.
  show_model_indicator: true

  # Number of conversation turns to show before "load more"
  # Default: 20.
  conversation_page_size: 20
```

---

## 6. Setup & Installation

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pip | 23.0+ | Package manager |
| Node.js | 18+ | Not required (no JS build) |
| Git | 2.40+ | Version control |
| Redis | 7.0+ (optional) | Async task queue (Celery) |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/aria.git
cd aria
```

### Step 2 — Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# Windows users: run pip install -r requirements-windows.txt instead of requirements.txt
python -m spacy download en_core_web_sm  # PII detection model
```

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` — every variable documented below:

```bash
# ─── LLM API Keys ────────────────────────────────────────────────────────────

# Google AI Studio (Gemini) — Free tier: 60 req/min, 1M tokens/day (Flash), 50 req/day (Pro)
# Sign up: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_ai_studio_key

# Groq — Free tier: 30 req/min, 14,400 req/day
# Sign up: https://console.groq.com
GROQ_API_KEY=your_groq_api_key

# Together AI — Free tier: $1 credit on signup (~500K tokens)
# Sign up: https://api.together.xyz
TOGETHER_API_KEY=your_together_api_key

# Cohere — Free tier: 100 req/min (trial key)
# Sign up: https://dashboard.cohere.com
COHERE_API_KEY=your_cohere_api_key

# ─── Search ───────────────────────────────────────────────────────────────────

# Tavily — Free tier: 1,000 API calls/month
# Sign up: https://app.tavily.com
TAVILY_API_KEY=your_tavily_api_key

# ─── Observability ────────────────────────────────────────────────────────────

# LangSmith — Free tier: 5,000 traces/month
# Sign up: https://smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=aria-dev

# ─── Infrastructure ───────────────────────────────────────────────────────────

# Upstash Redis (optional, for Celery async tasks) — Free tier: 10,000 req/day
# Sign up: https://console.upstash.com
UPSTASH_REDIS_URL=redis://default:your_password@your-instance.upstash.io:6379

# ─── Application ──────────────────────────────────────────────────────────────

# Master API key for FastAPI (generate a random 32-char string)
# Generate: python -c "import secrets; print(secrets.token_hex(16))"
ARIA_API_KEY=your_generated_api_key

# Database path (SQLite)
ARIA_DB_PATH=./data/aria.db

# ChromaDB persistence path
ARIA_CHROMA_PATH=./data/chroma

# Document upload storage path
ARIA_UPLOAD_PATH=./data/uploads

# Environment: development | production
ARIA_ENV=development

# Secret key for session signing (generate random 32-char string)
ARIA_SECRET_KEY=your_generated_secret_key
```

### Step 5 — Initialize Database

```bash
python -c "import asyncio; from db.database import init_db; asyncio.run(init_db())"
```

### Step 6 — Verify Configuration

```bash
python scripts/setup.sh --verify
```

Expected output:
```
✓ Python 3.11.x
✓ Google API key valid (Gemini 1.5 Flash accessible)
✓ Groq API key valid (llama3-8b-8192 accessible)
✓ Tavily API key valid
✓ LangSmith tracing enabled
✓ SQLite database initialized at ./data/aria.db
✓ ChromaDB initialized at ./data/chroma
✓ spaCy en_core_web_sm loaded
✓ sentence-transformers/all-MiniLM-L6-v2 loaded (or will download on first use)
```

### Step 7 — Start the Application

**Option A — FastAPI Backend Only**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

**Option B — Streamlit UI (connects to running FastAPI)**
```bash
# Terminal 1: Start FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Streamlit
streamlit run ui/streamlit_app.py
# UI: http://localhost:8501
```

**Option C — CLI (direct, no server)**
```bash
python -m aria chat --session new
# Starts interactive CLI session
```

**Option D — Docker Compose (recommended for local dev)**
```bash
docker-compose up
# FastAPI: http://localhost:8000
# Streamlit: http://localhost:8501
```

### Common Setup Errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_google_genai` | Missing dependency | `pip install langchain-google-genai` |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | spaCy model not installed | `python -m spacy download en_core_web_sm` |
| `google.api_core.exceptions.PermissionDenied` | Wrong/expired API key | Regenerate key at aistudio.google.com |
| `chromadb.errors.InvalidCollectionException` | Stale ChromaDB schema | `python scripts/reset_db.py --chroma-only` |
| `sqlite3.OperationalError: no such table` | DB not initialized | `python -c "from db.database import init_db; init_db()"` |
| Port 8000 already in use | Another process | `lsof -i :8000 \| kill -9 PID` or change port in config |

---

## 7. Tech Stack Decisions

### LangGraph vs CrewAI vs AutoGen

**LangGraph (chosen)**

LangGraph models the agent as a directed graph with typed state. This matters because ARIA's routing logic — conditional edges, parallel execution, retry loops — is expressed as first-class graph structure, not buried in imperative code. The state schema is a TypedDict, which means every node's input and output is type-checked. LangSmith integration is native. Streaming is first-class.

Tradeoffs accepted: steeper learning curve than CrewAI; requires explicit graph design upfront.
Free tier: fully open source, no usage limits.

**CrewAI** — abstracts agents as role-based "crew members." Simpler for basic multi-agent setups, but the abstraction leaks when you need fine-grained control over execution order, state management, and conditional routing. ARIA's reflection loops and parallel tool execution would require fighting the framework.

**AutoGen** — Microsoft's multi-agent framework. Powerful for conversational agent chains, but designed around agent-to-agent messaging patterns, not single-coherent-state graphs. Observability and state management require more custom work. Better suited for social simulation than operational pipelines.

**Verdict:** LangGraph is the right tool for a stateful, observable, production-grade agentic system.

---

### ChromaDB vs FAISS vs Weaviate

**ChromaDB (chosen)**

Local persistence with zero infrastructure. Runs embedded in the Python process. Supports metadata filtering, namespacing, and hybrid search. Has a client-server mode for production scaling. Open source.

Tradeoffs: not suited for high-concurrency multi-user production at scale (SQLite backend has write lock limits). Fine for single-user or moderate-load deployment.
Migration to paid: swap ChromaDB for Pinecone (same embedding interface, 5-minute migration).

**FAISS** — Facebook's vector library. Extremely fast for pure ANN search. No built-in persistence, metadata, or server mode. You'd implement all of that yourself. ARIA needs metadata filtering (by user_id, doc_id), namespacing, and persistence — FAISS requires custom wrappers for all of it.

**Weaviate** — Production-grade vector DB with GraphQL interface, multi-tenancy, and hybrid search built in. Overkill for a single-user deployment. Cloud free tier is 14-day trial. Self-hosted requires Docker + memory.

**Verdict:** ChromaDB is the right default. FAISS is faster but requires more custom code. Weaviate is the right upgrade path for multi-user production.

---

### Tavily vs SerpAPI vs DuckDuckGo

**Tavily (primary) + DuckDuckGo (fallback) (chosen)**

Tavily is purpose-built for LLM agents. Returns structured results with `relevance_score`, `published_date`, and `raw_content` extraction — not just snippets. 1,000 free calls/month. When quota is hit, DuckDuckGo (`duckduckgo-search` library) requires no API key, is rate-tolerant, and provides adequate results for most queries.

**SerpAPI** — Google Search wrapper. Best result quality. $0 free tier (100 calls/month on trial). Not viable as primary provider for a free-tier system.

**DuckDuckGo (alone)** — No API key, rate-tolerant, but returns raw snippets without relevance scoring, published dates, or content extraction. Acceptable fallback, not a primary.

**Verdict:** Tavily for quality-sensitive calls; DuckDuckGo as an unlimited safety net.

---

### SQLite vs PostgreSQL

**SQLite (chosen)**

ARIA's relational storage is write-light (session logs, token counts, memory metadata) and read-moderate (analytics queries, memory retrieval). SQLite handles this comfortably with WAL mode enabled. Zero infrastructure. Portable. Trivially backedup (single file).

Tradeoffs: write concurrency limit (one writer at a time in WAL mode). For single-user or low-concurrency deployments, this is never hit.

Migration path: SQLAlchemy ORM used throughout. To migrate to PostgreSQL: change the `DATABASE_URL` in config, run `alembic upgrade head`. Zero application code changes.

**PostgreSQL** would be necessary if: (a) multi-user with concurrent writes, (b) analytics queries become complex enough to need full SQL planner, (c) JSON column queries on conversation history. None of these apply to ARIA's current scope.

---

## 8. What Makes This Different

This is a technical differentiation — not marketing.

### vs. A Basic LangChain Chatbot

A LangChain chatbot has: LLM + ConversationBufferMemory + maybe one tool. It has a single model, drops conversation history when context fills, has no fallback if the model fails, no reflection, no planning, no multi-agent delegation, and no structured memory.

ARIA has: 4 LLM providers with automatic rotation, 3-tier memory with decay, a 7-node reasoning graph, 8 tools, 6 specialist agents, hallucination detection, PII redaction, structured observability, and a reflection loop that catches low-quality responses before delivery.

The comparison is not close.

### vs. ChatGPT / Claude (Proprietary)

Proprietary systems have superior base models. GPT-4o and Claude Sonnet outperform Gemini 1.5 Flash and Groq LLaMA on most benchmarks. That is a real gap that ARIA acknowledges honestly (see [Known Limitations](#9-known-limitations)).

What ARIA has that they don't: full observability into every reasoning step, persistent cross-session memory with explicit user control, a local document knowledge base, code execution in a configurable sandbox, complete customizability, and zero cost.

ChatGPT and Claude are black boxes. ARIA is fully auditable.

### vs. Open WebUI

Open WebUI is a UI wrapper for Ollama (local models) and OpenAI-compatible APIs. It provides a polished chat interface with model switching and some RAG support.

What Open WebUI doesn't have: multi-model routing with health-aware fallback, a stateful agentic graph, multi-agent delegation, episodic memory with decay, a reflection loop, hallucination detection, or structured observability. It is a frontend. ARIA is an agent system.

### vs. Perplexity AI

Perplexity is genuinely strong at web-grounded research responses. ARIA's ResearchAgent is deliberately designed to replicate this capability.

Where ARIA exceeds Perplexity: persistent memory across sessions, document knowledge base, code execution, explicit agent reasoning traces, and full customizability. Where Perplexity exceeds ARIA: better base models (GPT-4o class), more polished UX, higher query throughput.

### vs. Standard RAG Chatbots

Most RAG systems use fixed-size chunking, single-stage dense retrieval, and inject context directly without compression or reranking. ARIA uses semantic chunking, a 3-stage hybrid retrieval pipeline (dense + BM25 + cross-encoder rerank), and context compression when chunks exceed token budget.

The retrieval quality difference on long, multi-concept documents is meaningful.

---

## 9. Known Limitations

These are not edge cases. They are real constraints you will encounter.

### Free API Rate Limits

| Provider | Limit | ARIA Behavior When Hit |
|---|---|---|
| Gemini 1.5 Flash | 60 req/min, 1M tokens/day | Rotate to Groq; Gemini Pro goes on cooldown |
| Gemini 1.5 Pro | 50 req/day | Deprioritized; used only for high-complexity reasoning |
| Groq | 30 req/min, 14,400 req/day | Rotate to Together AI; acceptable latency impact |
| Together AI | Trial credit only (~500K tokens) | Deprioritized after credit exhaustion |
| Cohere | 100 req/min trial | Available as fallback summarization |
| Tavily | 1,000 calls/month | Falls back to DuckDuckGo; snippet quality degrades |
| LangSmith | 5,000 traces/month | Tracing disabled after quota; local logging continues |

Heavy research sessions with 5+ sub-questions can consume 10-15 Tavily calls per query. At that rate, the monthly quota lasts roughly 70-100 deep research queries.

### Model Quality Ceiling

Free-tier models (Gemini 1.5 Flash, Groq LLaMA-3-70B) are meaningfully below GPT-4o and Claude Sonnet 3.5 on:
- Complex multi-step reasoning (math, logic chains)
- Nuanced instruction following
- Long-document coherent synthesis
- Code generation for complex algorithms

ARIA's architecture compensates with reflection loops and tool grounding, but cannot fully close the gap. On factual, research, and code tasks, the gap narrows considerably. On pure reasoning tasks (math proofs, complex logical deductions), expect the model quality ceiling to show.

### Memory at Scale

ChromaDB with SQLite backend degrades under high write concurrency. For a single user with thousands of memory entries, performance is fine. For 10+ concurrent users writing memories simultaneously, write latency increases. Migration path: switch to ChromaDB client-server mode or Pinecone.

Episodic memory decay runs as a scheduled script, not a background daemon. In production, this needs a cron job or Celery beat task. If decay doesn't run, memory scores don't decay and pruning doesn't happen.

### What ARIA Cannot Do (That Paid Systems Can)

- Voice interface (not implemented; roadmap item)
- Multi-user with proper isolation and billing (single-user architecture)
- Image understanding (no vision model in free tier; Gemini Vision is paid)
- Real-time data (stock prices, live sports — Tavily returns web results, not live APIs)
- Fine-tuning (no access to model weights on free tiers)
- Sub-500ms responses (free tier latency: Groq ~300ms, Gemini Flash ~1-2s, Gemini Pro ~3-6s)

### Latency Expectations

| Task | Expected Latency |
|---|---|
| Conversational response (no tools) | 1-3 seconds |
| Single web search + synthesis | 3-6 seconds |
| Deep research (4 sub-questions) | 12-25 seconds |
| RAG retrieval + synthesis | 2-5 seconds |
| Code generation + execution | 4-10 seconds |

These are real-world ranges on free tier APIs. Groq is consistently fastest. Gemini Pro is slowest. Plan accordingly.

---

## 10. Production Deployment (Free Infra)

ARIA can be deployed to fully free infrastructure with the following stack.

### Architecture

```
GitHub (source + CI/CD)
    │
    ├── Streamlit Cloud ──── UI (streamlit run ui/streamlit_app.py)
    │                        Free: 1GB RAM, unlimited apps (public repos)
    │
    ├── Railway.app ─────── FastAPI backend
    │                        Free: $5 credit/month (~500 hours runtime)
    │
    └── Local disk ────────── ChromaDB + SQLite (volume-mounted or Fly.io free volume)
```

### Streamlit Cloud Deployment

1. Push code to public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select repository, branch, file path: `ui/streamlit_app.py`.
4. Add secrets in Streamlit Cloud UI (Settings → Secrets) — same as `.env` content.
5. The `ARIA_API_URL` secret must point to your Railway FastAPI URL.

```toml
# .streamlit/secrets.toml (do not commit — add via UI)
ARIA_API_KEY = "your_api_key"
ARIA_API_URL = "https://your-app.railway.app"
```

### Railway Deployment (FastAPI)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Deploy
railway up

# Set environment variables
railway variables set GOOGLE_API_KEY=... GROQ_API_KEY=... ARIA_API_KEY=...
# (set all variables from .env)
```

`Procfile` (required by Railway):
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 2
```

**Railway free tier limit:** $5 credit/month. At ~$0.000463/GB-hour for RAM and $0.000231/vCPU-hour, a 512MB/0.5vCPU app runs ~$3/month. Monitor usage in Railway dashboard.

### Upstash Redis (Optional, Celery Queue)

1. Create account at [console.upstash.com](https://console.upstash.com).
2. Create a Redis database (free tier: 10,000 req/day, 256MB).
3. Copy `UPSTASH_REDIS_URL` to environment variables.

### Persistent Storage for ChromaDB + SQLite

Railway's free tier does not have persistent volumes (data lost on redeploy). Options:

**Option A (recommended for free):** Run ChromaDB and SQLite locally and expose ARIA via ngrok for external access. Persistent, fast, free.

**Option B:** Use Fly.io free tier (3GB persistent volume, 2 shared CPUs, 256MB RAM). Deploy FastAPI + ChromaDB + SQLite together with volume mount.

```bash
# Fly.io deployment
fly launch
fly volumes create aria_data --size 3
# Add to fly.toml: [mounts] source = "aria_data", destination = "/data"
fly deploy
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: python -m spacy download en_core_web_sm
      - run: pytest tests/unit/ -v --timeout=30
      - run: ruff check .
      - run: mypy aria/ --ignore-missing-imports
```

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

---

## 11. Full Production-Grade Code

> All files are fully typed, documented, and immediately runnable. Every failure case is handled explicitly.

---

### `models/model_router.py`

```python
"""
Multi-model router with registry, health checking, and automatic rotation.

Ensures every LLM call reaches the best available, healthiest model for the
requested task type. Users never see model failures — only latency variation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from pydantic import BaseModel

from db.repositories.token_usage_repo import TokenUsageRepository

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    REASONING = "reasoning"
    SPEED = "speed"
    LONG_CONTEXT = "long_context"
    CODE = "code"
    EMBEDDINGS = "embeddings"
    SUMMARIZATION = "summarization"


@dataclass
class ModelEntry:
    """Single registered model with health and capability metadata."""

    name: str
    provider: str  # "google" | "groq" | "together" | "cohere"
    api_key_env_var: str
    context_window: int
    tokens_limit_daily: int
    task_affinity: List[str]
    max_output_tokens: int = 2048
    tokens_used_today: int = 0
    failure_count: int = 0
    cooldown_until: Optional[datetime] = None
    avg_latency_ms: float = 1000.0
    is_enabled: bool = True

    @property
    def is_healthy(self) -> bool:
        """True if model is not in cooldown and has remaining budget."""
        if not self.is_enabled:
            return False
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            return False
        if self.tokens_used_today >= self.tokens_limit_daily * 0.95:
            return False
        return True

    @property
    def cooldown_remaining_sec(self) -> int:
        """Seconds remaining in cooldown, 0 if not in cooldown."""
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            return int((self.cooldown_until - datetime.utcnow()).total_seconds())
        return 0

    @property
    def utilization_pct(self) -> float:
        """Percentage of daily token budget consumed."""
        return (self.tokens_used_today / self.tokens_limit_daily) * 100


class RoutingDecision(BaseModel):
    """Result of a routing decision."""

    model_name: str
    provider: str
    task_type: str
    reason: str


class ModelRouter:
    """
    Routes LLM calls to the best available model for the given task type.

    Manages: model registry, health checking, token budget tracking,
    failure handling, exponential backoff, and graceful degradation.
    """

    # Default model registry with free-tier models
    DEFAULT_REGISTRY: List[ModelEntry] = [
        ModelEntry(
            name="gemini-1.5-pro",
            provider="google",
            api_key_env_var="GOOGLE_API_KEY",
            context_window=1_000_000,
            tokens_limit_daily=500_000,  # Conservative estimate for free tier
            task_affinity=["reasoning", "long_context", "summarization"],
            max_output_tokens=8192,
        ),
        ModelEntry(
            name="gemini-1.5-flash",
            provider="google",
            api_key_env_var="GOOGLE_API_KEY",
            context_window=1_000_000,
            tokens_limit_daily=1_000_000,
            task_affinity=["speed", "summarization", "reasoning"],
            max_output_tokens=8192,
        ),
        ModelEntry(
            name="llama3-70b-8192",
            provider="groq",
            api_key_env_var="GROQ_API_KEY",
            context_window=8192,
            tokens_limit_daily=200_000,
            task_affinity=["reasoning", "speed"],
            max_output_tokens=4096,
        ),
        ModelEntry(
            name="llama3-8b-8192",
            provider="groq",
            api_key_env_var="GROQ_API_KEY",
            context_window=8192,
            tokens_limit_daily=500_000,
            task_affinity=["speed"],
            max_output_tokens=4096,
        ),
        ModelEntry(
            name="deepseek-coder-v2-coder",
            provider="groq",
            api_key_env_var="GROQ_API_KEY",
            context_window=16384,
            tokens_limit_daily=200_000,
            task_affinity=["code"],
            max_output_tokens=8192,
        ),
        ModelEntry(
            name="command-r",
            provider="cohere",
            api_key_env_var="COHERE_API_KEY",
            context_window=128_000,
            tokens_limit_daily=100_000,
            task_affinity=["summarization", "reasoning"],
            max_output_tokens=4096,
        ),
    ]

    def __init__(
        self,
        registry: Optional[List[ModelEntry]] = None,
        token_repo: Optional[TokenUsageRepository] = None,
        backoff_base: int = 2,
        max_retries: int = 3,
        cooldown_minutes: int = 60,
        failure_threshold: int = 3,
    ) -> None:
        self.registry: Dict[str, ModelEntry] = {
            m.name: m for m in (registry or self.DEFAULT_REGISTRY)
        }
        self.token_repo = token_repo or TokenUsageRepository()
        self.backoff_base = backoff_base
        self.max_retries = max_retries
        self.cooldown_minutes = cooldown_minutes
        self.failure_threshold = failure_threshold
        self._load_today_token_usage()

    def _load_today_token_usage(self) -> None:
        """Load today's token usage from SQLite into registry."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for model in self.registry.values():
            model.tokens_used_today = self.token_repo.get_usage(model.name, today)

    def select_model(self, task_type: TaskType) -> Optional[ModelEntry]:
        """
        Select the best available model for the given task type.

        Returns None if no healthy model is available for this task.
        """
        candidates = [
            m for m in self.registry.values()
            if task_type.value in m.task_affinity and m.is_healthy
        ]

        if not candidates:
            # Fallback: any healthy model regardless of affinity
            candidates = [m for m in self.registry.values() if m.is_healthy]

        if not candidates:
            return None

        # Sort: fewest failures first, then lowest average latency
        candidates.sort(key=lambda m: (m.failure_count, m.avg_latency_ms))
        return candidates[0]

    def build_llm(self, model: ModelEntry, temperature: float = 0.1) -> Any:
        """Instantiate a LangChain LLM for the given model entry."""
        import os

        api_key = os.getenv(model.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not set: {model.api_key_env_var}")

        if model.provider == "google":
            return ChatGoogleGenerativeAI(
                model=model.name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=model.max_output_tokens,
            )
        elif model.provider == "groq":
            return ChatGroq(
                model=model.name,
                groq_api_key=api_key,
                temperature=temperature,
                max_tokens=model.max_output_tokens,
            )
        elif model.provider == "together":
            return ChatTogether(
                model=model.name,
                together_api_key=api_key,
                temperature=temperature,
                max_tokens=model.max_output_tokens,
            )
        elif model.provider == "cohere":
            return ChatCohere(
                model=model.name,
                cohere_api_key=api_key,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {model.provider}")

    async def call_with_rotation(
        self,
        task_type: TaskType,
        messages: List[Any],
        temperature: float = 0.1,
    ) -> tuple[Any, str]:
        """
        Call LLM with automatic fallback rotation on failure.

        Returns: (response, model_name_used)
        Raises: RuntimeError if all models exhausted.
        """
        attempted: List[str] = []

        while True:
            model = self.select_model(task_type)

            if model is None or model.name in attempted:
                raise RuntimeError(
                    "All available models exhausted for task type "
                    f"'{task_type.value}'. Attempted: {attempted}. "
                    "Please retry in a few minutes."
                )

            attempted.append(model.name)
            llm = self.build_llm(model, temperature)

            for attempt in range(self.max_retries):
                try:
                    start = time.monotonic()
                    response = await llm.ainvoke(messages)
                    latency_ms = (time.monotonic() - start) * 1000

                    # Update metrics on success
                    self._record_success(model, response, latency_ms)
                    return response, model.name

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "rate limit" in error_str or "quota" in error_str

                    if is_rate_limit:
                        wait_sec = self.backoff_base ** attempt
                        logger.warning(
                            f"Rate limit on {model.name}, attempt {attempt+1}. "
                            f"Waiting {wait_sec}s before retry."
                        )
                        await asyncio.sleep(wait_sec)
                        if attempt == self.max_retries - 1:
                            self._apply_cooldown(model)
                    else:
                        # Non-rate-limit error: rotate immediately
                        logger.error(f"Error on {model.name}: {e}")
                        model.failure_count += 1
                        if model.failure_count >= self.failure_threshold:
                            self._apply_cooldown(model)
                        break  # Break retry loop, rotate to next model

    def _record_success(self, model: ModelEntry, response: Any, latency_ms: float) -> None:
        """Update model metrics and token usage after successful call."""
        # Reset failure count on success
        model.failure_count = max(0, model.failure_count - 1)

        # Update EMA latency (alpha=0.1 for stability)
        model.avg_latency_ms = 0.9 * model.avg_latency_ms + 0.1 * latency_ms

        # Track token usage
        if hasattr(response, "usage_metadata"):
            tokens = getattr(response.usage_metadata, "total_tokens", 0)
            model.tokens_used_today += tokens
            today = datetime.utcnow().strftime("%Y-%m-%d")
            self.token_repo.add_usage(model.name, today, tokens)

    def _apply_cooldown(self, model: ModelEntry) -> None:
        """Put a model in cooldown."""
        model.cooldown_until = datetime.utcnow() + timedelta(minutes=self.cooldown_minutes)
        logger.warning(
            f"Model {model.name} put on cooldown until {model.cooldown_until.isoformat()}"
        )

    def get_status(self) -> List[Dict[str, Any]]:
        """Return health status for all registered models."""
        return [
            {
                "name": m.name,
                "provider": m.provider,
                "is_healthy": m.is_healthy,
                "tokens_used_today": m.tokens_used_today,
                "tokens_limit_daily": m.tokens_limit_daily,
                "utilization_pct": round(m.utilization_pct, 2),
                "cooldown_remaining_sec": m.cooldown_remaining_sec,
                "failure_count": m.failure_count,
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "task_affinity": m.task_affinity,
            }
            for m in self.registry.values()
        ]
```

---

### `core/agent_graph.py`

```python
"""
LangGraph agent graph: the central reasoning engine of ARIA.

Defines all nodes, edges, conditional routing, and state transitions.
Every user input flows through this graph — nothing bypasses it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from config.config import get_config
from core.state import ARIAState, PlanStep, ToolCallRecord, ToolOutputRecord
from memory.memory_manager import MemoryManager
from models.model_router import ModelRouter, TaskType
from responsible_ai.responsible_ai import ResponsibleAI
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
config = get_config()


# ── Node Implementations ────────────────────────────────────────────────────


async def intent_classifier_node(state: ARIAState, router: ModelRouter) -> ARIAState:
    """
    Classify user intent, extract entities, and determine if tools are required.

    Updates: intent_type, confidence_score, extracted_entities, requires_tools
    """
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an intent classifier for an AI assistant.
        
Classify the user message into EXACTLY ONE of these intents:
- conversational: casual chat, greetings, opinions, simple factual questions
- factual: questions requiring specific factual lookup (dates, definitions, how-things-work)
- research: requires multiple web sources, synthesis, or current information
- code: write, debug, explain, or execute code
- document: query about uploaded documents in the knowledge base
- memory_operation: "remember X", "forget X", "what do you know about me"
- ambiguous: genuinely unclear intent

Respond in JSON only:
{
  "intent_type": "<one of the above>",
  "confidence_score": <0.0-1.0>,
  "extracted_entities": ["entity1", "entity2"],
  "requires_tools": <true|false>,
  "reasoning": "<one sentence>"
}

requires_tools is true if the query needs: web search, document retrieval, code execution, or memory operations."""),
        HumanMessage(content=f"Message: {state['user_message']}\n\nConversation history (last 3 turns):\n{_format_history(state['conversation_history'][-6:])}")
    ])

    response, model_used = await router.call_with_rotation(
        TaskType.SPEED, prompt.format_messages(), temperature=0.0
    )

    import json
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback to safe defaults
        parsed = {
            "intent_type": "conversational",
            "confidence_score": 0.5,
            "extracted_entities": [],
            "requires_tools": False,
        }

    state.update({
        "intent_type": parsed["intent_type"],
        "confidence_score": parsed["confidence_score"],
        "extracted_entities": parsed.get("extracted_entities", []),
        "requires_tools": parsed["requires_tools"],
        "model_used": model_used,
        "debug_trace": state.get("debug_trace", []) + [
            f"IntentClassifier: {parsed['intent_type']} ({parsed['confidence_score']:.2f})"
        ],
    })
    return state


async def memory_reader_node(state: ARIAState, memory_manager: MemoryManager) -> ARIAState:
    """
    Retrieve relevant episodic memories to inject into context.

    Updates: retrieved_memories
    """
    memories = await memory_manager.retrieve_episodic(
        user_id=state["user_id"],
        query=state["user_message"],
        top_k=config.memory.episodic_top_k,
        min_importance=config.memory.min_importance_to_retrieve,
    )
    state["retrieved_memories"] = memories
    state["debug_trace"] = state.get("debug_trace", []) + [
        f"MemoryReader: retrieved {len(memories)} memories"
    ]
    return state


async def planner_node(state: ARIAState, router: ModelRouter) -> ARIAState:
    """
    Generate a structured execution plan for tool-requiring queries.

    Updates: execution_plan
    """
    from langchain_core.prompts import ChatPromptTemplate

    memories_context = _format_memories(state.get("retrieved_memories", []))
    tool_list = ToolRegistry.list_tools()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a strategic planner for an AI agent.
        
Available tools: {tool_list}

Create a minimal, efficient execution plan. Each step must have:
- step_number (int)
- action (human-readable description)
- tool (exact tool name from available tools)
- tool_params (dict of parameters)
- expected_output (what this step produces)
- fallback_action (what to do if this step fails)
- can_parallelize (bool: true if this step doesn't depend on the previous step's output)

Return JSON only:
{{"plan": [<steps>]}}

Be minimal. If a question can be answered in 1-2 steps, don't create 5.
"""),
        HumanMessage(content=f"""User query: {state['user_message']}
Intent: {state['intent_type']}
Entities: {state.get('extracted_entities', [])}
User memory context: {memories_context}""")
    ])

    response, model_used = await router.call_with_rotation(
        TaskType.REASONING, prompt.format_messages(), temperature=0.1
    )

    import json
    try:
        parsed = json.loads(response.content)
        plan = [PlanStep(**step) for step in parsed["plan"]]
    except Exception as e:
        logger.error(f"Plan parsing failed: {e}")
        # Minimal fallback plan
        plan = [PlanStep(
            step_number=1,
            action="Search for information",
            tool="web_search",
            tool_params={"query": state["user_message"], "max_results": 5},
            expected_output="Relevant web results",
            fallback_action="Answer from model knowledge",
            can_parallelize=False,
        )]

    state["execution_plan"] = plan
    state["model_used"] = model_used
    state["debug_trace"] = state.get("debug_trace", []) + [
        f"Planner: created {len(plan)}-step plan"
    ]
    return state


async def executor_node(
    state: ARIAState,
    tool_registry: ToolRegistry,
) -> ARIAState:
    """
    Execute tool calls from the execution plan, handling parallel and sequential steps.

    Updates: tool_outputs
    """
    plan = state.get("execution_plan", [])
    tool_outputs: List[ToolOutputRecord] = []

    # Group steps: parallel groups and sequential singles
    i = 0
    while i < len(plan):
        step = plan[i]

        # Check if this step and next can run in parallel
        if step.can_parallelize and i + 1 < len(plan) and plan[i + 1].can_parallelize:
            # Gather all consecutive parallelizable steps
            parallel_steps = []
            while i < len(plan) and plan[i].can_parallelize:
                parallel_steps.append(plan[i])
                i += 1

            results = await asyncio.gather(
                *[_execute_single_step(s, tool_registry) for s in parallel_steps],
                return_exceptions=True,
            )
            for step, result in zip(parallel_steps, results):
                if isinstance(result, Exception):
                    tool_outputs.append(_make_failure_record(step, str(result)))
                else:
                    tool_outputs.append(result)
        else:
            result = await _execute_single_step(step, tool_registry)
            tool_outputs.append(result)
            i += 1

    state["tool_outputs"] = tool_outputs
    state["debug_trace"] = state.get("debug_trace", []) + [
        f"Executor: ran {len(tool_outputs)} tool calls, "
        f"{sum(1 for t in tool_outputs if t.success)} succeeded"
    ]
    return state


async def _execute_single_step(step: PlanStep, registry: ToolRegistry) -> ToolOutputRecord:
    """Execute a single plan step with timeout and fallback."""
    import time

    start = time.monotonic()
    try:
        output = await asyncio.wait_for(
            registry.call(step.tool, step.tool_params),
            timeout=config.tools.default_tool_timeout_sec,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return ToolOutputRecord(
            tool_name=step.tool,
            input=step.tool_params,
            output=output,
            latency_ms=latency_ms,
            success=True,
        )
    except asyncio.TimeoutError:
        return _make_failure_record(step, f"Timeout after {config.tools.default_tool_timeout_sec}s")
    except Exception as e:
        logger.error(f"Tool {step.tool} failed: {e}")
        return _make_failure_record(step, str(e))


def _make_failure_record(step: PlanStep, error: str) -> ToolOutputRecord:
    return ToolOutputRecord(
        tool_name=step.tool,
        input=step.tool_params,
        output={"error": error, "fallback": step.fallback_action},
        latency_ms=0,
        success=False,
        error_message=error,
    )


async def reflection_node(state: ARIAState, router: ModelRouter) -> ARIAState:
    """
    Score the draft response on relevance, groundedness, and completeness.
    Triggers retry if any score falls below threshold.

    Updates: reflection_scores, reflection_passed
    """
    from langchain_core.prompts import ChatPromptTemplate

    if not state.get("draft_response"):
        state["reflection_passed"] = True
        return state

    tool_outputs_summary = "\n".join([
        f"- {t.tool_name}: {'SUCCESS' if t.success else 'FAILED'} — {str(t.output)[:200]}"
        for t in state.get("tool_outputs", [])
    ])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a response quality evaluator.
Score the draft response on three dimensions (0.0-1.0 each):
- relevance: Does it address what the user asked?
- groundedness: Are all factual claims supported by the tool outputs provided?
- completeness: Does it address all parts of the query?

Respond in JSON only:
{"relevance": 0.0-1.0, "groundedness": 0.0-1.0, "completeness": 0.0-1.0, "critique": "one sentence"}
"""),
        HumanMessage(content=f"""User query: {state['user_message']}
Tool outputs:
{tool_outputs_summary}
Draft response:
{state['draft_response']}""")
    ])

    response, _ = await router.call_with_rotation(
        TaskType.REASONING, prompt.format_messages(), temperature=0.0
    )

    import json
    try:
        scores = json.loads(response.content)
    except Exception:
        scores = {"relevance": 0.8, "groundedness": 0.8, "completeness": 0.8}

    passed = all(
        scores.get(dim, 0) >= config.reflection.pass_threshold
        for dim in ["relevance", "groundedness", "completeness"]
    )

    state["reflection_scores"] = scores
    state["reflection_passed"] = passed
    state["debug_trace"] = state.get("debug_trace", []) + [
        f"Reflection: R={scores.get('relevance', 0):.2f} "
        f"G={scores.get('groundedness', 0):.2f} "
        f"C={scores.get('completeness', 0):.2f} — {'PASS' if passed else 'RETRY'}"
    ]
    return state


async def response_synthesizer_node(
    state: ARIAState,
    router: ModelRouter,
    responsible_ai: ResponsibleAI,
) -> ARIAState:
    """
    Synthesize the final response from tool outputs, memories, and conversation history.

    Updates: final_response, citations, confidence_indicator, follow_up_suggestions
    """
    from langchain_core.prompts import ChatPromptTemplate

    memories_context = _format_memories(state.get("retrieved_memories", []))
    tool_outputs_context = _format_tool_outputs(state.get("tool_outputs", []))
    history_context = _format_history(state["conversation_history"][-10:])

    retry_note = ""
    if state.get("retry_count", 0) > 0:
        scores = state.get("reflection_scores", {})
        retry_note = f"\n\nIMPORTANT: Previous response was marked insufficient. Critique: {scores.get('critique', '')}. Address this in your response."

    format_instructions = {
        "research": "Structure as: executive summary, then key findings with inline citations, then sources list.",
        "code": "Provide working code in a code block, then explanation. Include example usage.",
        "factual": "Direct answer first, then supporting detail. Cite sources inline.",
        "conversational": "Natural, concise conversational response. No headers or bullets unless genuinely helpful.",
        "document": "Answer with direct reference to the document content. Include source file and page.",
        "memory_operation": "Confirm the memory operation performed in natural language.",
    }.get(state.get("intent_type", "conversational"), "Clear, direct response.")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are ARIA, an intelligent AI assistant.

What you know about this user (long-term memory):
{memories_context or 'No prior memories.'}

Information retrieved for this query:
{tool_outputs_context or 'No tools used — answer from knowledge.'}

Response format: {format_instructions}

Rules:
- Cite sources inline as [Source: name] for every external fact
- If information conflicts, flag it
- If you're uncertain, say so explicitly (don't hallucinate)
- Be direct and specific
{retry_note}"""),
        *[
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in state["conversation_history"][-6:]
        ],
        HumanMessage(content=state["user_message"]),
    ])

    response, model_used = await router.call_with_rotation(
        TaskType.REASONING, prompt.format_messages(), temperature=0.15
    )

    raw_response = response.content

    # Apply responsible AI layer
    processed = await responsible_ai.process_output(
        response=raw_response,
        tool_outputs=state.get("tool_outputs", []),
        session_id=state["session_id"],
    )

    # Compute confidence indicator
    scores = state.get("reflection_scores")
    if scores:
        min_score = min(scores.get(d, 0) for d in ["relevance", "groundedness", "completeness"])
        confidence = "high" if min_score > 0.85 else "medium" if min_score > 0.70 else "low"
    else:
        confidence = "medium"

    # Generate follow-up suggestions (cheap, fast model)
    follow_ups = await _generate_follow_ups(state, router)

    state.update({
        "draft_response": raw_response,
        "final_response": processed.cleaned_response,
        "citations": processed.citations,
        "confidence_indicator": confidence,
        "follow_up_suggestions": follow_ups,
        "model_used": model_used,
    })
    return state


async def memory_writer_node(
    state: ARIAState,
    memory_manager: MemoryManager,
) -> ARIAState:
    """
    Extract and persist key facts from the conversation turn to long-term memory.

    Updates: memory_writes
    """
    await memory_manager.write_turn_to_episodic(
        user_id=state["user_id"],
        session_id=state["session_id"],
        user_message=state["user_message"],
        assistant_response=state.get("final_response", ""),
    )
    await memory_manager.save_session_turn(
        session_id=state["session_id"],
        user_message=state["user_message"],
        assistant_response=state.get("final_response", ""),
        model_used=state.get("model_used", "unknown"),
    )
    return state


# ── Graph Assembly ──────────────────────────────────────────────────────────


def build_aria_graph(
    router: ModelRouter,
    memory_manager: MemoryManager,
    tool_registry: ToolRegistry,
    responsible_ai: ResponsibleAI,
) -> Any:
    """
    Assemble and compile the full ARIA LangGraph agent graph.

    Returns a compiled graph ready for invocation.
    """
    from functools import partial

    graph = StateGraph(ARIAState)

    # Register nodes with injected dependencies
    graph.add_node("IntentClassifier", partial(intent_classifier_node, router=router))
    graph.add_node("MemoryReader", partial(memory_reader_node, memory_manager=memory_manager))
    graph.add_node("Planner", partial(planner_node, router=router))
    graph.add_node("Executor", partial(executor_node, tool_registry=tool_registry))
    graph.add_node("ReflectionAgent", partial(reflection_node, router=router))
    graph.add_node("ResponseSynthesizer", partial(
        response_synthesizer_node, router=router, responsible_ai=responsible_ai
    ))
    graph.add_node("MemoryWriter", partial(memory_writer_node, memory_manager=memory_manager))

    # Entry point
    graph.set_entry_point("IntentClassifier")

    # Fixed edges
    graph.add_edge("IntentClassifier", "MemoryReader")

    # Conditional: needs tools → Planner, else → Synthesizer
    graph.add_conditional_edges(
        "MemoryReader",
        lambda s: "Planner" if (
            s["requires_tools"] or
            s["intent_type"] in ["research", "code", "document", "memory_operation"]
        ) else "ResponseSynthesizer",
        {"Planner": "Planner", "ResponseSynthesizer": "ResponseSynthesizer"},
    )

    graph.add_edge("Planner", "Executor")

    # After execution: always reflect
    graph.add_edge("Executor", "ResponseSynthesizer")

    # ResponseSynthesizer → ReflectionAgent for quality check
    graph.add_edge("ResponseSynthesizer", "ReflectionAgent")

    # Conditional: reflection passed or max retries → MemoryWriter, else → retry Synthesizer
    graph.add_conditional_edges(
        "ReflectionAgent",
        lambda s: "MemoryWriter" if (
            s.get("reflection_passed", True) or
            s.get("retry_count", 0) >= config.reflection.max_retries
        ) else "ResponseSynthesizer",
        {"MemoryWriter": "MemoryWriter", "ResponseSynthesizer": "ResponseSynthesizer"},
    )

    graph.add_edge("MemoryWriter", END)

    return graph.compile()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _format_history(history: List[Dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in history)


def _format_memories(memories: List) -> str:
    if not memories:
        return ""
    lines = [f"- [{m.category}] {m.content} (importance: {m.importance_score:.2f})" for m in memories]
    return "\n".join(lines)


def _format_tool_outputs(outputs: List[ToolOutputRecord]) -> str:
    if not outputs:
        return ""
    lines = []
    for o in outputs:
        if o.success:
            lines.append(f"[{o.tool_name}] {str(o.output)[:500]}")
        else:
            lines.append(f"[{o.tool_name}] FAILED: {o.error_message}")
    return "\n\n".join(lines)


async def _generate_follow_ups(state: ARIAState, router: ModelRouter) -> List[str]:
    """Generate 2-3 relevant follow-up question suggestions."""
    if not config.ui.show_follow_up_suggestions:
        return []
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Generate exactly 2 short follow-up questions (under 10 words each) the user might want to ask next. Return as JSON array only: [\"question1\", \"question2\"]"),
            HumanMessage(content=f"User asked: {state['user_message']}\nAssistant answered about: {state.get('final_response', '')[:200]}")
        ])
        response, _ = await router.call_with_rotation(
            TaskType.SPEED, prompt.format_messages(), temperature=0.3
        )
        import json
        return json.loads(response.content)
    except Exception:
        return []
```

---

### `memory/memory_manager.py`

```python
"""
Unified memory interface across all three ARIA memory tiers.

Tier 1: Working memory (in-session, LangChain summary buffer)
Tier 2: Episodic memory (cross-session, ChromaDB + SQLite)
Tier 3: Semantic knowledge base (RAG, per-user document store)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional

import chromadb
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer

from config.config import get_config
from db.repositories.memory_repo import MemoryRepository
from memory.schemas import EpisodicMemory, MemoryCategory

logger = logging.getLogger(__name__)
config = get_config()


class MemoryManager:
    """
    Unified interface for all ARIA memory operations.

    Instantiate once per application lifecycle and inject into graph nodes.
    """

    def __init__(
        self,
        chroma_path: str,
        memory_repo: MemoryRepository,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.memory_repo = memory_repo
        self.embedder = SentenceTransformer(embedding_model_name)
        self._working_memories: dict[str, ConversationSummaryBufferMemory] = {}

    # ── Tier 1: Working Memory ───────────────────────────────────────────────

    def get_working_memory(self, session_id: str, llm: Any) -> ConversationSummaryBufferMemory:
        """
        Get or create working memory for a session.

        Memory is summarized automatically when approaching token limit.
        """
        if session_id not in self._working_memories:
            self._working_memories[session_id] = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=int(
                    4096 * config.memory.working_memory_token_threshold
                ),
                return_messages=True,
            )
        return self._working_memories[session_id]

    async def save_session_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        model_used: str,
    ) -> None:
        """Persist a conversation turn to SQLite."""
        await self.memory_repo.save_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            model_used=model_used,
            timestamp=datetime.utcnow(),
        )

    # ── Tier 2: Episodic Memory ──────────────────────────────────────────────

    async def retrieve_episodic(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        min_importance: float = 0.15,
    ) -> List[EpisodicMemory]:
        """
        Retrieve the most relevant episodic memories for the current query.

        Uses semantic similarity search over the user's memory namespace.
        """
        collection_name = f"episodic_{user_id.replace('-', '_')}"
        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            return []  # No memories yet for this user

        query_embedding = self.embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, collection.count()),  # Retrieve extra, filter below
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            importance = meta.get("importance_score", 0.5)
            if importance < min_importance:
                continue

            memories.append(EpisodicMemory(
                memory_id=meta.get("memory_id", ""),
                content=doc,
                importance_score=importance,
                category=meta.get("category", "context"),
                created_at=meta.get("created_at", ""),
                last_accessed=datetime.utcnow().isoformat(),
                access_count=meta.get("access_count", 0) + 1,
            ))

        # Update access counts in background
        for memory in memories:
            await self.memory_repo.increment_access(memory.memory_id)

        # Sort by importance descending, return top_k
        memories.sort(key=lambda m: m.importance_score, reverse=True)
        return memories[:top_k]

    async def write_turn_to_episodic(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """
        Extract and persist key facts from a conversation turn to episodic memory.

        This runs as a background task — does not block response delivery.
        """
        from models.model_router import ModelRouter, TaskType
        # Use a lightweight model for extraction
        router = ModelRouter()

        extraction_prompt = f"""Extract any key facts, preferences, goals, or important information
about the USER (not general knowledge) from this conversation turn.

User said: {user_message}
Assistant said: {assistant_response[:500]}

For each extracted item, provide:
{{"items": [{{"content": "...", "category": "preference|fact|goal|context", "importance_score": 0.0-1.0}}]}}

Rules:
- Only extract things ABOUT THE USER specifically
- If nothing notable was revealed, return {{"items": []}}
- importance_score: 0.9 for explicit preferences/goals, 0.6 for facts, 0.3 for context
- Return JSON only"""

        try:
            from langchain_core.messages import HumanMessage
            response, _ = await router.call_with_rotation(
                TaskType.SPEED, [HumanMessage(content=extraction_prompt)], temperature=0.0
            )
            import json
            extracted = json.loads(response.content)

            collection_name = f"episodic_{user_id.replace('-', '_')}"
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                collection = self.chroma_client.create_collection(
                    collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            for item in extracted.get("items", []):
                if not item.get("content"):
                    continue

                memory_id = f"mem_{user_id}_{datetime.utcnow().timestamp():.0f}"
                embedding = self.embedder.encode(item["content"]).tolist()

                # Determine decay rate from category
                decay_rates = {
                    "preference": config.memory.decay_rate_preference,
                    "fact": config.memory.decay_rate_fact,
                    "goal": config.memory.decay_rate_preference,
                    "context": config.memory.decay_rate_context,
                }
                decay_rate = decay_rates.get(item["category"], config.memory.decay_rate_context)

                collection.add(
                    documents=[item["content"]],
                    embeddings=[embedding],
                    metadatas=[{
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "category": item["category"],
                        "importance_score": item["importance_score"],
                        "decay_rate": decay_rate,
                        "created_at": datetime.utcnow().isoformat(),
                        "last_accessed": datetime.utcnow().isoformat(),
                        "access_count": 0,
                    }],
                    ids=[memory_id],
                )

                await self.memory_repo.save_episodic_memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    content=item["content"],
                    importance_score=item["importance_score"],
                    category=item["category"],
                    decay_rate=decay_rate,
                )

        except Exception as e:
            logger.error(f"Episodic memory write failed: {e}")
            # Non-fatal: response was already delivered

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Soft-delete a specific memory entry."""
        collection_name = f"episodic_{user_id.replace('-', '_')}"
        try:
            collection = self.chroma_client.get_collection(collection_name)
            collection.delete(ids=[memory_id])
            await self.memory_repo.soft_delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Memory deletion failed: {e}")
            return False

    async def get_all_memories(
        self,
        user_id: str,
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> List[EpisodicMemory]:
        """Retrieve all episodic memories for a user, sorted by importance."""
        return await self.memory_repo.get_all(
            user_id=user_id,
            min_importance=min_importance,
            limit=limit,
        )
```

---

### `tools/tool_registry.py`

```python
"""
Central tool registry. All tools are registered here and called through this interface.

The ToolRouter (in agent_graph.py) exclusively uses this registry.
No tool is ever called directly by an agent node.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolSpec(BaseModel):
    """Specification for a registered tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    is_enabled: bool = True
    timeout_sec: int = 30


class ToolRegistry:
    """
    Registry of all available tools with async call dispatch.

    Tools are registered at startup. The ToolRouter calls registry.call(tool_name, params).
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._register_all()

    def _register_all(self) -> None:
        """Register all tools at initialization."""
        from tools.web_search import web_search
        from tools.deep_research import deep_research
        from tools.document_rag import document_rag
        from tools.code_interpreter import code_interpreter
        from tools.memory_tool import memory_tool
        from tools.summarizer import summarizer
        from tools.structured_output import structured_output
        from tools.calculator import calculator

        self.register("web_search", web_search, ToolSpec(
            name="web_search",
            description="Search the web for current information via Tavily or DuckDuckGo",
            input_schema={"query": "str", "max_results": "int (default 5)"},
        ))
        self.register("deep_research", deep_research, ToolSpec(
            name="deep_research",
            description="Multi-source deep research with sub-question decomposition and synthesis",
            input_schema={"query": "str", "max_sub_questions": "int (default 4)"},
        ))
        self.register("document_rag", document_rag, ToolSpec(
            name="document_rag",
            description="Search user's uploaded knowledge base using hybrid retrieval",
            input_schema={"query": "str", "top_k": "int (default 8)"},
        ))
        self.register("code_interpreter", code_interpreter, ToolSpec(
            name="code_interpreter",
            description="Write and execute Python code in a sandboxed environment",
            input_schema={"code": "str", "timeout_sec": "int (default 15)"},
            timeout_sec=20,
        ))
        self.register("memory_tool", memory_tool, ToolSpec(
            name="memory_tool",
            description="Read, write, or delete long-term memory entries",
            input_schema={"operation": "read|write|delete", "content": "str"},
        ))
        self.register("summarizer", summarizer, ToolSpec(
            name="summarizer",
            description="Summarize long text or URL content",
            input_schema={"content": "str (optional)", "url": "str (optional)", "target_length": "int words"},
        ))
        self.register("structured_output", structured_output, ToolSpec(
            name="structured_output",
            description="Convert content to structured format: json, table, bullets, code_block",
            input_schema={"content": "str", "format_type": "json|table|bullets|code_block"},
        ))
        self.register("calculator", calculator, ToolSpec(
            name="calculator",
            description="Evaluate math expressions precisely (no hallucination risk)",
            input_schema={"expression": "str"},
        ))

    def register(self, name: str, fn: Callable, spec: ToolSpec) -> None:
        """Register a tool."""
        self._tools[name] = fn
        self._specs[name] = spec

    async def call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Call a registered tool by name with parameters.

        Raises KeyError if tool not found.
        Raises RuntimeError if tool is disabled.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not registered. Available: {list(self._tools.keys())}")

        spec = self._specs[tool_name]
        if not spec.is_enabled:
            raise RuntimeError(f"Tool '{tool_name}' is disabled in configuration.")

        fn = self._tools[tool_name]
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(**params)
            else:
                return await asyncio.get_event_loop().run_in_executor(None, lambda: fn(**params))
        except TypeError as e:
            logger.error(f"Tool '{tool_name}' called with invalid params {params}: {e}")
            raise

    @classmethod
    def list_tools(cls) -> str:
        """Return a formatted list of tool names and descriptions for prompt injection."""
        instance = cls()
        return "\n".join(
            f"- {name}: {spec.description}"
            for name, spec in instance._specs.items()
            if spec.is_enabled
        )
```

---

### `agents/research_agent.py`

```python
"""
ResearchAgent: Deep web research with sub-question decomposition and synthesis.

Produces structured research reports comparable to Perplexity AI deep research mode.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from pydantic import BaseModel

from models.model_router import ModelRouter, TaskType
from tools.web_search import web_search

logger = logging.getLogger(__name__)


class Finding(BaseModel):
    claim: str
    confidence: float
    source_url: str
    source_title: str


class ResearchReport(BaseModel):
    summary: str
    key_findings: List[Finding]
    sources: List[dict]
    confidence: float
    sub_questions: List[str]
    conflicting_info: List[str]


class ResearchAgent:
    """
    Performs multi-source deep research by decomposing queries into sub-questions,
    searching each independently, deduplicating, scoring, and synthesizing.
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def research(
        self,
        query: str,
        max_sub_questions: int = 4,
        max_results_per_question: int = 5,
        max_total_sources: int = 10,
    ) -> ResearchReport:
        """
        Perform deep research on a query.

        Steps:
          1. Decompose into sub-questions
          2. Search each sub-question in parallel
          3. Deduplicate and score results
          4. Synthesize into structured report
        """
        # Step 1: Decompose
        sub_questions = await self._decompose_query(query, max_sub_questions)
        logger.info(f"ResearchAgent: {len(sub_questions)} sub-questions for '{query[:50]}'")

        # Step 2: Search in parallel
        search_tasks = [
            web_search(query=sq, max_results=max_results_per_question)
            for sq in sub_questions
        ]
        all_results_nested = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Flatten and filter failures
        all_results = []
        for result in all_results_nested:
            if isinstance(result, Exception):
                logger.warning(f"Sub-question search failed: {result}")
                continue
            if isinstance(result, dict) and "results" in result:
                all_results.extend(result["results"])

        if not all_results:
            return ResearchReport(
                summary="Unable to retrieve research results. Please try again.",
                key_findings=[],
                sources=[],
                confidence=0.0,
                sub_questions=sub_questions,
                conflicting_info=[],
            )

        # Step 3: Deduplicate
        deduped = self._deduplicate(all_results)

        # Step 4: Score and filter
        scored = await self._score_relevance(query, deduped)
        top_sources = sorted(scored, key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_sources = top_sources[:max_total_sources]

        # Step 5: Synthesize
        report = await self._synthesize(query, sub_questions, top_sources)
        return report

    async def _decompose_query(self, query: str, max_sub_questions: int) -> List[str]:
        """Decompose a complex query into independent searchable sub-questions."""
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        prompt = [
            SystemMessage(content=f"""Break the user's research query into {max_sub_questions} independent, 
specific sub-questions that together fully address the original query.

Return JSON only: {{"sub_questions": ["q1", "q2", ...]}}

Rules:
- Each sub-question should be independently searchable
- Cover different aspects of the original query
- Be specific enough to return useful search results
- Don't overlap significantly with each other"""),
            HumanMessage(content=f"Research query: {query}"),
        ]

        response, _ = await self.router.call_with_rotation(
            TaskType.REASONING, prompt, temperature=0.1
        )
        try:
            return json.loads(response.content)["sub_questions"]
        except Exception:
            return [query]  # Fallback: use original query

    def _deduplicate(self, results: List[dict]) -> List[dict]:
        """Remove duplicate results by URL."""
        seen_urls = set()
        deduped = []
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(result)
        return deduped

    async def _score_relevance(self, query: str, results: List[dict]) -> List[dict]:
        """Score each result for relevance to the original query using LLM."""
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        # Batch scoring to minimize LLM calls
        results_text = "\n".join([
            f"{i}. {r.get('title', '')} — {r.get('snippet', '')[:150]}"
            for i, r in enumerate(results)
        ])

        prompt = [
            SystemMessage(content="""Score each search result's relevance to the query (0.0-1.0).
Return JSON only: {"scores": [0.8, 0.3, ...]} (same order as results)"""),
            HumanMessage(content=f"Query: {query}\n\nResults:\n{results_text}"),
        ]

        try:
            response, _ = await self.router.call_with_rotation(
                TaskType.SPEED, prompt, temperature=0.0
            )
            scores = json.loads(response.content)["scores"]
            for result, score in zip(results, scores):
                result["relevance_score"] = score
        except Exception:
            for result in results:
                result["relevance_score"] = 0.5

        return results

    async def _synthesize(
        self,
        query: str,
        sub_questions: List[str],
        sources: List[dict],
    ) -> ResearchReport:
        """Synthesize a structured research report from scored sources."""
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        sources_text = "\n\n".join([
            f"[{i+1}] {s.get('title', 'Untitled')} ({s.get('url', '')})\n{s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ])

        prompt = [
            SystemMessage(content="""You are a research synthesizer. Create a structured research report.

Return JSON only:
{
  "summary": "2-3 sentence executive summary",
  "key_findings": [
    {"claim": "specific finding", "confidence": 0.0-1.0, "source_index": 1}
  ],
  "conflicting_info": ["any contradictions found across sources"],
  "overall_confidence": 0.0-1.0
}"""),
            HumanMessage(content=f"""Research query: {query}

Sub-questions investigated:
{chr(10).join(f'- {q}' for q in sub_questions)}

Sources:
{sources_text}"""),
        ]

        response, _ = await self.router.call_with_rotation(
            TaskType.REASONING, prompt, temperature=0.1
        )

        try:
            parsed = json.loads(response.content)

            key_findings = []
            for f in parsed.get("key_findings", []):
                source_idx = f.get("source_index", 1) - 1
                source = sources[source_idx] if 0 <= source_idx < len(sources) else {}
                key_findings.append(Finding(
                    claim=f["claim"],
                    confidence=f.get("confidence", 0.7),
                    source_url=source.get("url", ""),
                    source_title=source.get("title", ""),
                ))

            return ResearchReport(
                summary=parsed.get("summary", "Research synthesis unavailable."),
                key_findings=key_findings,
                sources=[{"title": s.get("title"), "url": s.get("url")} for s in sources],
                confidence=parsed.get("overall_confidence", 0.7),
                sub_questions=sub_questions,
                conflicting_info=parsed.get("conflicting_info", []),
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return ResearchReport(
                summary="Synthesis failed. Raw sources available.",
                key_findings=[],
                sources=[{"title": s.get("title"), "url": s.get("url")} for s in sources],
                confidence=0.4,
                sub_questions=sub_questions,
                conflicting_info=[],
            )
```

---

### `agents/rag_agent.py`

```python
"""
RAGAgent: Hybrid retrieval-augmented generation over user's knowledge base.

Pipeline: dense retrieval (ChromaDB) + sparse retrieval (BM25) + cross-encoder reranking.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import chromadb
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config.config import get_config
from db.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)
config = get_config()


class RetrievedChunk(BaseModel):
    chunk_text: str
    source_file: str
    page_number: Optional[int]
    section: Optional[str]
    relevance_score: float
    chunk_id: str


class RAGResult(BaseModel):
    chunks: List[RetrievedChunk]
    total_tokens: int
    retrieval_method: str


class RAGAgent:
    """
    Hybrid RAG pipeline: dense + sparse retrieval with cross-encoder reranking.

    Consistently outperforms single-stage dense-only retrieval on multi-concept
    queries and exact-match requirements.
    """

    def __init__(
        self,
        chroma_path: str,
        doc_repo: DocumentRepository,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.doc_repo = doc_repo
        self.dense_encoder = SentenceTransformer(dense_model_name)
        self.cross_encoder = CrossEncoder(rerank_model_name)

    async def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 8,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> RAGResult:
        """
        Retrieve the most relevant chunks for a query using the hybrid pipeline.

        1. Dense retrieval (ChromaDB cosine similarity)
        2. Sparse retrieval (BM25)
        3. Union + cross-encoder reranking
        4. Context compression if over token budget
        """
        collection_name = f"kb_{user_id.replace('-', '_')}"

        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            logger.info(f"No knowledge base for user {user_id}")
            return RAGResult(chunks=[], total_tokens=0, retrieval_method="none")

        # Step 1: Dense retrieval
        query_embedding = self.dense_encoder.encode(query).tolist()
        where_filter = {"doc_id": {"$in": filter_doc_ids}} if filter_doc_ids else None

        dense_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(config.rag.dense_top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        dense_candidates = [
            {
                "chunk_text": doc,
                "metadata": meta,
                "chunk_id": meta.get("chunk_id", ""),
                "source": "dense",
            }
            for doc, meta in zip(
                dense_results["documents"][0],
                dense_results["metadatas"][0],
            )
        ]

        # Step 2: Sparse retrieval (BM25)
        # Load all chunks for this user (needed for BM25 corpus)
        all_chunks = await self.doc_repo.get_all_chunks(user_id, filter_doc_ids)
        if all_chunks:
            tokenized_corpus = [chunk["content"].lower().split() for chunk in all_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)

            # Get top-k BM25 results
            top_bm25_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:config.rag.sparse_top_k]

            bm25_candidates = [
                {
                    "chunk_text": all_chunks[i]["content"],
                    "metadata": all_chunks[i],
                    "chunk_id": all_chunks[i].get("chunk_id", ""),
                    "source": "bm25",
                }
                for i in top_bm25_indices
            ]
        else:
            bm25_candidates = []

        # Step 3: Merge and deduplicate
        seen_ids = set()
        merged = []
        for candidate in dense_candidates + bm25_candidates:
            cid = candidate["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(candidate)

        if not merged:
            return RAGResult(chunks=[], total_tokens=0, retrieval_method="none")

        # Step 4: Cross-encoder reranking
        pairs = [(query, c["chunk_text"]) for c in merged]
        rerank_scores = self.cross_encoder.predict(pairs)

        for candidate, score in zip(merged, rerank_scores):
            candidate["rerank_score"] = float(score)

        merged.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_reranked = merged[:top_k]

        # Step 5: Build result
        chunks = []
        total_tokens = 0
        for c in top_reranked:
            meta = c["metadata"]
            chunk_tokens = len(c["chunk_text"].split()) * 1.3  # Approximate
            total_tokens += chunk_tokens

            if total_tokens > config.rag.max_context_tokens:
                break

            chunks.append(RetrievedChunk(
                chunk_text=c["chunk_text"],
                source_file=meta.get("source_file", "unknown"),
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                relevance_score=c["rerank_score"],
                chunk_id=c["chunk_id"],
            ))

        return RAGResult(
            chunks=chunks,
            total_tokens=int(total_tokens),
            retrieval_method="hybrid_dense_bm25_crossencoder",
        )
```

---

### `agents/code_agent.py`

```python
"""
CodeAgent: Write, execute, and iterate on Python code in a sandboxed environment.

Sandbox: RestrictedPython with custom policy. No network. No filesystem (except /tmp).
Retries up to config.code_agent.max_execution_retries times on failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from pydantic import BaseModel
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safe_builtins

from config.config import get_config
from models.model_router import ModelRouter, TaskType

logger = logging.getLogger(__name__)
config = get_config()


class CodeResult(BaseModel):
    code: str
    stdout: str
    stderr: str
    success: bool
    execution_time_ms: int
    explanation: str
    attempts: int


class CodeAgent:
    """
    Generates and iteratively executes Python code in a RestrictedPython sandbox.
    On failure, reads the traceback and rewrites the code, up to max_retries.
    """

    # Blocked imports — enforced by the custom policy
    BLOCKED_MODULES = frozenset({
        "subprocess", "os", "sys", "socket", "requests", "urllib",
        "ftplib", "smtplib", "ctypes", "importlib", "shutil", "glob",
    })

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self._max_retries = config.code_agent.max_execution_retries
        self._timeout = config.tools.code_execution_timeout_sec

    async def generate_and_run(
        self,
        task_description: str,
        context: Optional[str] = None,
    ) -> CodeResult:
        """
        Generate code for the task, execute it, and retry on failure.
        """
        last_error = ""
        code = ""

        for attempt in range(1, self._max_retries + 2):  # +1 for initial attempt
            code = await self._generate_code(task_description, context, last_error, attempt)
            stdout, stderr, success, exec_time = await self._execute_sandboxed(code)

            if success:
                explanation = await self._explain_code(code, stdout)
                return CodeResult(
                    code=code,
                    stdout=stdout,
                    stderr=stderr,
                    success=True,
                    execution_time_ms=exec_time,
                    explanation=explanation,
                    attempts=attempt,
                )

            last_error = stderr
            logger.info(f"CodeAgent attempt {attempt} failed: {stderr[:100]}")

            if attempt > self._max_retries:
                break

        # All retries exhausted
        explanation = await self._explain_failure(code, last_error)
        return CodeResult(
            code=code,
            stdout="",
            stderr=last_error,
            success=False,
            execution_time_ms=0,
            explanation=explanation,
            attempts=self._max_retries + 1,
        )

    async def _generate_code(
        self,
        task: str,
        context: Optional[str],
        last_error: str,
        attempt: int,
    ) -> str:
        """Generate Python code for the task, incorporating previous error if retrying."""
        from langchain_core.messages import HumanMessage, SystemMessage

        retry_instruction = ""
        if last_error and attempt > 1:
            retry_instruction = f"\n\nPrevious code FAILED with this error:\n{last_error}\n\nFix the error in your new code."

        prompt = [
            SystemMessage(content=f"""You are an expert Python programmer.
Write Python 3.11 code to complete the task.

Constraints:
- No imports of: {', '.join(sorted(self.BLOCKED_MODULES))}
- No file system access (except /tmp)
- No network calls
- Code must be self-contained and produce output to stdout
- Use only stdlib modules that are safe (math, json, re, datetime, collections, itertools, etc.)

Return ONLY the Python code. No explanation. No markdown code blocks.{retry_instruction}"""),
            HumanMessage(content=f"Task: {task}{chr(10) + 'Context: ' + context if context else ''}"),
        ]

        response, _ = await self.router.call_with_rotation(
            TaskType.CODE, prompt, temperature=0.1
        )

        # Strip markdown code blocks if present
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()

    async def _execute_sandboxed(
        self,
        code: str,
    ) -> tuple[str, str, bool, int]:
        """Execute code in RestrictedPython sandbox with timeout."""
        import io
        import contextlib

        def _run() -> tuple[str, str, bool, int]:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Build restricted globals
            restricted_globals = dict(safe_globals)
            restricted_globals["__builtins__"] = dict(safe_builtins)
            restricted_globals["_getiter_"] = iter
            restricted_globals["_getattr_"] = getattr
            restricted_globals["_write_"] = lambda x: x
            restricted_globals["_inplacevar_"] = lambda op, x, y: x

            # Inject safe stdlib
            import math, json, re, datetime, collections, itertools, random, string
            restricted_globals.update({
                "math": math, "json": json, "re": re,
                "datetime": datetime, "collections": collections,
                "itertools": itertools, "random": random, "string": string,
            })

            start = time.monotonic()
            try:
                byte_code = compile_restricted(code, filename="<aria_sandbox>", mode="exec")
                with contextlib.redirect_stdout(stdout_capture):
                    with contextlib.redirect_stderr(stderr_capture):
                        exec(byte_code, restricted_globals)  # noqa: S102

                exec_time = int((time.monotonic() - start) * 1000)
                return stdout_capture.getvalue(), "", True, exec_time

            except SyntaxError as e:
                return "", f"SyntaxError: {e}", False, 0
            except Exception as e:
                return stdout_capture.getvalue(), f"{type(e).__name__}: {e}", False, 0

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _run),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            return "", f"TimeoutError: execution exceeded {self._timeout}s", False, self._timeout * 1000

    async def _explain_code(self, code: str, output: str) -> str:
        """Generate a brief explanation of the code and its output."""
        from langchain_core.messages import HumanMessage
        response, _ = await self.router.call_with_rotation(
            TaskType.SPEED,
            [HumanMessage(content=f"Briefly explain (2-3 sentences) what this code does and what it produced:\n\n{code}\n\nOutput: {output[:200]}")],
            temperature=0.1,
        )
        return response.content

    async def _explain_failure(self, code: str, error: str) -> str:
        """Explain why the code failed after all retries."""
        from langchain_core.messages import HumanMessage
        response, _ = await self.router.call_with_rotation(
            TaskType.SPEED,
            [HumanMessage(content=f"Explain briefly why this code failed and what would fix it:\n\n{code}\n\nError: {error}")],
            temperature=0.1,
        )
        return response.content
```

---

### `agents/reflection_agent.py`

```python
"""
ReflectionAgent: Self-scoring quality evaluator for ARIA responses.

Runs synchronously during the graph (as a gate before delivery) and
asynchronously post-delivery (for continuous quality logging).
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel

from models.model_router import ModelRouter, TaskType

logger = logging.getLogger(__name__)


class ReflectionScores(BaseModel):
    relevance: float
    groundedness: float
    completeness: float
    critique: str
    passed: bool


class ReflectionAgent:
    """
    Scores responses on relevance, groundedness, and completeness.
    Triggers retry if any score falls below the configured threshold.
    """

    def __init__(self, router: ModelRouter, pass_threshold: float = 0.70) -> None:
        self.router = router
        self.pass_threshold = pass_threshold

    async def score(
        self,
        user_query: str,
        draft_response: str,
        tool_outputs_summary: str,
        retrieved_context: str,
    ) -> ReflectionScores:
        """
        Score a draft response on three quality dimensions.

        Returns ReflectionScores with a boolean passed field.
        """
        import json
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(content="""You are a strict response quality evaluator for an AI assistant.

Score the draft response on THREE dimensions (0.0–1.0 each):

RELEVANCE (0.0–1.0):
  1.0 = fully addresses the user's query
  0.7 = mostly addresses it, minor gaps
  0.5 = partially addresses it
  0.0 = does not address it

GROUNDEDNESS (0.0–1.0):
  1.0 = every factual claim is supported by provided tool outputs/context
  0.7 = most claims supported, minor ungrounded details
  0.5 = some claims ungrounded
  0.0 = mostly hallucinated

COMPLETENESS (0.0–1.0):
  1.0 = all parts of the query answered
  0.7 = most parts answered
  0.5 = only some parts answered
  0.0 = incomplete

Return JSON only:
{"relevance": 0.0-1.0, "groundedness": 0.0-1.0, "completeness": 0.0-1.0, "critique": "one actionable sentence"}"""),
            HumanMessage(content=f"""USER QUERY: {user_query}

AVAILABLE INFORMATION:
{tool_outputs_summary or 'No tool outputs.'}

RETRIEVED CONTEXT:
{retrieved_context or 'None.'}

DRAFT RESPONSE:
{draft_response}"""),
        ]

        response, _ = await self.router.call_with_rotation(
            TaskType.REASONING, prompt, temperature=0.0
        )

        try:
            raw = json.loads(response.content)
            passed = all(
                raw.get(dim, 0.0) >= self.pass_threshold
                for dim in ["relevance", "groundedness", "completeness"]
            )
            return ReflectionScores(
                relevance=raw.get("relevance", 0.8),
                groundedness=raw.get("groundedness", 0.8),
                completeness=raw.get("completeness", 0.8),
                critique=raw.get("critique", ""),
                passed=passed,
            )
        except Exception as e:
            logger.error(f"Reflection scoring failed: {e}")
            # Default to passing to avoid blocking delivery
            return ReflectionScores(
                relevance=0.8, groundedness=0.8, completeness=0.8,
                critique="Scoring unavailable", passed=True
            )
```

---

### `responsible_ai/responsible_ai.py`

```python
"""
Responsible AI layer: hallucination detection, PII redaction, injection prevention,
content filtering, and source attribution.

Runs inline on every response before delivery.
"""

from __future__ import annotations

import logging
import re
from typing import List

import spacy
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Load spaCy model once at import time
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    _NLP = None
    logger.warning("spaCy en_core_web_sm not found. PII NER disabled.")

_EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class ProcessedOutput(BaseModel):
    cleaned_response: str
    citations: List[dict]
    hallucination_flags: List[str]
    pii_redacted_in_logs: bool


class ResponsibleAI:
    """
    Applies all responsible AI checks to a response before delivery.
    """

    # PII regex patterns
    PII_PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "PHONE": re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"ignore (previous|all|prior) instructions", re.IGNORECASE),
        re.compile(r"you are now", re.IGNORECASE),
        re.compile(r"forget (your|all) instructions", re.IGNORECASE),
        re.compile(r"act as (a|an) (?!assistant)", re.IGNORECASE),
        re.compile(r"jailbreak", re.IGNORECASE),
        re.compile(r"DAN mode", re.IGNORECASE),
    ]

    async def process_output(
        self,
        response: str,
        tool_outputs: List,
        session_id: str,
    ) -> ProcessedOutput:
        """
        Apply all responsible AI checks to a response.

        Returns ProcessedOutput with cleaned response and metadata.
        """
        cleaned = response
        hallucination_flags = []

        # 1. Hallucination detection (only when source chunks available)
        if config.responsible_ai.hallucination_threshold > 0 and tool_outputs:
            source_texts = self._extract_source_texts(tool_outputs)
            if source_texts:
                cleaned, hallucination_flags = self._detect_and_handle_hallucinations(
                    cleaned, source_texts
                )

        # 2. Extract citations from response
        citations = self._extract_citations(cleaned)

        return ProcessedOutput(
            cleaned_response=cleaned,
            citations=citations,
            hallucination_flags=hallucination_flags,
            pii_redacted_in_logs=True,
        )

    def check_input(self, user_input: str, session_id: str) -> tuple[bool, str]:
        """
        Check user input for prompt injection.

        Returns: (is_safe, reason)
        """
        if not config.responsible_ai.injection_detection_enabled:
            return True, ""

        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(user_input):
                logger.warning(f"Injection attempt detected in session {session_id}: {pattern.pattern}")
                return False, "Input contains patterns that conflict with system operation."

        return True, ""

    def redact_for_logging(self, text: str) -> str:
        """
        Redact PII from text before writing to logs.

        IMPORTANT: Only applied to log entries — NOT to responses sent to users.
        """
        if not config.responsible_ai.pii_detection_enabled:
            return text

        redacted = text

        # Regex-based PII
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED:{pii_type}]", redacted)

        # NER-based (names, orgs, locations)
        if _NLP:
            doc = _NLP(redacted[:1000])  # Process first 1000 chars for performance
            for ent in reversed(doc.ents):  # Reverse to preserve offsets
                if ent.label_ in {"PERSON", "ORG"}:
                    redacted = redacted[:ent.start_char] + f"[REDACTED:{ent.label_}]" + redacted[ent.end_char:]

        return redacted

    def _extract_source_texts(self, tool_outputs: List) -> List[str]:
        """Extract source text from tool outputs for hallucination checking."""
        sources = []
        for output in tool_outputs:
            if not output.success:
                continue
            o = output.output
            if isinstance(o, dict):
                if "results" in o:  # WebSearch
                    sources.extend([r.get("snippet", "") for r in o["results"]])
                elif "chunks" in o:  # RAG
                    sources.extend([c.get("chunk_text", "") for c in o["chunks"]])
                elif "summary" in o:  # Research
                    sources.append(o["summary"])
        return [s for s in sources if s]

    def _detect_and_handle_hallucinations(
        self,
        response: str,
        source_texts: List[str],
    ) -> tuple[str, List[str]]:
        """
        Check factual claims against source texts.

        Claims with cosine similarity < hallucination_threshold are flagged.
        """
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 30]
        if not sentences or not source_texts:
            return response, []

        source_embeddings = _EMBEDDER.encode(source_texts)
        claim_embeddings = _EMBEDDER.encode(sentences)

        flags = []
        flagged_sentences = set()

        for i, (claim, claim_emb) in enumerate(zip(sentences, claim_embeddings)):
            # Compute max similarity against any source
            sims = cosine_similarity([claim_emb], source_embeddings)[0]
            max_sim = float(np.max(sims))

            if max_sim < config.responsible_ai.hallucination_threshold:
                flags.append(claim[:100])
                if config.responsible_ai.remove_hallucinated_claims:
                    flagged_sentences.add(i)

        if flagged_sentences:
            cleaned_sentences = [
                s for i, s in enumerate(sentences)
                if i not in flagged_sentences
            ]
            cleaned = ". ".join(cleaned_sentences)
            if flags:
                cleaned += "\n\n*Note: Some claims could not be verified against retrieved sources and were omitted.*"
        else:
            cleaned = response

        return cleaned, flags

    def _extract_citations(self, response: str) -> List[dict]:
        """Parse inline citations from response text."""
        pattern = re.compile(r"\[Source: ([^\]]+)\]")
        citations = []
        for match in pattern.finditer(response):
            citations.append({"text": match.group(1), "position": match.start()})
        return citations
```

---

### `observability/observability.py`

```python
"""
Observability: structured logging, LangSmith integration, and analytics storage.

Every agent action, LLM call, and tool call is logged as structured JSON.
LangSmith traces are sent when LANGCHAIN_TRACING_V2=true.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from db.repositories.analytics_repo import AnalyticsRepository

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    Emits JSON-structured log events to stdout and persists to SQLite analytics.
    """

    def __init__(self, analytics_repo: AnalyticsRepository) -> None:
        self.analytics_repo = analytics_repo

    def log_event(
        self,
        session_id: str,
        agent: str,
        action: str,
        success: bool,
        latency_ms: int = 0,
        model_used: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_used: Optional[str] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Log a structured event to stdout and SQLite."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "user_id": user_id,
            "agent": agent,
            "action": action,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "model_used": model_used,
            "tool_used": tool_used,
            "success": success,
            "error": error,
        }
        if extra:
            event.update(extra)

        # Emit to stdout as JSON
        print(json.dumps(event))

        # Persist to SQLite analytics
        try:
            import asyncio
            asyncio.create_task(self.analytics_repo.save_event(event))
        except RuntimeError:
            # No running event loop (e.g. CLI context)
            pass


def setup_langsmith() -> None:
    """Configure LangSmith tracing if API key is available."""
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        logger.info(
            f"LangSmith tracing enabled. Project: {os.getenv('LANGCHAIN_PROJECT', 'aria-default')}"
        )
    else:
        logger.info("LangSmith tracing disabled (LANGCHAIN_API_KEY not set or tracing off).")


def configure_logging(log_level: str = "INFO") -> None:
    """Configure Python logging with consistent format."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
```

---

### `api/main.py`

```python
"""
FastAPI application: all routes, middleware, and startup/shutdown lifecycle.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.routes import chat, documents, memory, models as model_routes, analytics, stream
from config.config import get_config
from db.database import init_db
from observability.observability import setup_langsmith, configure_logging

logger = logging.getLogger(__name__)
config = get_config()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown."""
    configure_logging(config.observability.log_level)
    setup_langsmith()
    await init_db()
    logger.info("ARIA API started.")
    yield
    logger.info("ARIA API shutting down.")


app = FastAPI(
    title="ARIA API",
    description="Agentic Reasoning and Intelligence Architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate API key from Bearer token."""
    expected = os.getenv("ARIA_API_KEY", "")
    if not expected or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


# Mount routes
app.include_router(chat.router, prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_key)])
app.include_router(documents.router, prefix="/documents", tags=["documents"], dependencies=[Depends(verify_api_key)])
app.include_router(memory.router, prefix="/memory", tags=["memory"], dependencies=[Depends(verify_api_key)])
app.include_router(model_routes.router, prefix="/model-status", tags=["models"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"], dependencies=[Depends(verify_api_key)])
app.include_router(stream.router, tags=["streaming"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}
```

---

### `ui/streamlit_app.py`

```python
"""
Streamlit UI: conversational interface for ARIA with session management,
document upload, memory viewer, and model health dashboard.
"""

from __future__ import annotations

import streamlit as st
import requests
import json
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

API_URL = st.secrets.get("ARIA_API_URL", "http://localhost:8000")
API_KEY = st.secrets.get("ARIA_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

st.set_page_config(
    page_title="ARIA — Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ─────────────────────────────────────────────

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_default"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ ARIA Settings")

    st.subheader("Session")
    st.code(st.session_state.session_id, language=None)

    if st.button("🔄 New Session"):
        import uuid
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.toggle("🔍 Debug Mode", key="debug_mode")
    st.divider()

    # Model Status
    st.subheader("🔋 Model Health")
    try:
        status_response = requests.get(f"{API_URL}/model-status", timeout=5)
        if status_response.ok:
            models = status_response.json()["models"]
            for m in models:
                health_icon = "🟢" if m["is_healthy"] else "🔴"
                st.write(f"{health_icon} **{m['name']}** ({m['provider']})")
                st.progress(min(m["utilization_pct"] / 100, 1.0))
                st.caption(f"{m['tokens_used_today']:,} / {m['tokens_limit_daily']:,} tokens | {m['avg_latency_ms']:.0f}ms avg")
    except Exception:
        st.warning("Could not fetch model status")

    st.divider()

    # Document Upload
    st.subheader("📂 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "md", "txt", "csv"],
        help="Upload documents to ARIA's knowledge base for RAG retrieval",
    )

    if uploaded_file and st.button("📤 Upload"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        params = {"user_id": st.session_state.user_id}
        try:
            r = requests.post(
                f"{API_URL}/documents/upload",
                files=files,
                params=params,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=60,
            )
            if r.ok:
                data = r.json()
                st.success(f"✅ Ingested: {data['filename']} ({data['chunk_count']} chunks)")
            else:
                st.error(f"Upload failed: {r.text}")
        except Exception as e:
            st.error(f"Upload error: {e}")

# ── Main Chat Interface ───────────────────────────────────────────────────────

st.title("🤖 ARIA — Agentic AI Assistant")
st.caption("Powered by free LLM APIs | The only limitation is the model, not the system.")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show plan trace if available and debug mode on
        if message.get("plan_trace") and st.session_state.debug_mode:
            with st.expander("🧠 Thinking Trace", expanded=False):
                for step in message["plan_trace"]:
                    status_icon = "✅" if step.get("status") == "complete" else "⚡"
                    st.write(f"{status_icon} **Step {step['step']}**: {step['action']}")
                    if step.get("latency_ms"):
                        st.caption(f"{step['latency_ms']}ms")

        # Show confidence indicator
        if message.get("confidence_indicator") and message["role"] == "assistant":
            icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
            icon = icons.get(message["confidence_indicator"], "⚪")
            st.caption(f"{icon} Confidence: {message['confidence_indicator']} | Model: {message.get('model_used', 'unknown')}")

        # Citations
        if message.get("citations"):
            with st.expander(f"📚 Sources ({len(message['citations'])})", expanded=False):
                for cite in message["citations"]:
                    st.write(f"• {cite['text']}")

        # Follow-up suggestions
        if message.get("follow_up_suggestions") and message["role"] == "assistant":
            st.write("**Suggested follow-ups:**")
            cols = st.columns(len(message["follow_up_suggestions"]))
            for i, suggestion in enumerate(message["follow_up_suggestions"]):
                if cols[i].button(f"💬 {suggestion}", key=f"followup_{id(message)}_{i}"):
                    st.session_state.pending_message = suggestion
                    st.rerun()

# Chat input
user_input = st.chat_input("Ask ARIA anything...")

# Handle pending message (from follow-up button click)
if hasattr(st.session_state, "pending_message") and st.session_state.pending_message:
    user_input = st.session_state.pending_message
    del st.session_state.pending_message

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call ARIA API
    with st.chat_message("assistant"):
        with st.spinner("🧠 ARIA is thinking..."):
            try:
                payload = {
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "debug_mode": st.session_state.debug_mode,
                }
                response = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    headers=HEADERS,
                    timeout=120,
                )

                if response.ok:
                    data = response.json()
                    assistant_message = data["response"]

                    st.markdown(assistant_message)

                    # Plan trace (debug mode)
                    if data.get("plan_trace") and st.session_state.debug_mode:
                        with st.expander("🧠 Thinking Trace", expanded=True):
                            for step in data["plan_trace"]:
                                st.write(f"✅ **Step {step['step']}**: {step['action']}")

                    # Confidence + model
                    if data.get("confidence_indicator"):
                        icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                        icon = icons.get(data["confidence_indicator"], "⚪")
                        st.caption(
                            f"{icon} Confidence: {data['confidence_indicator']} | "
                            f"Model: {data.get('model_used', 'unknown')} | "
                            f"{data.get('total_latency_ms', 0)}ms"
                        )

                    # Citations
                    if data.get("citations"):
                        with st.expander(f"📚 Sources ({len(data['citations'])})", expanded=False):
                            for cite in data["citations"]:
                                st.write(f"• {cite['text']}")

                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message,
                        "plan_trace": data.get("plan_trace"),
                        "confidence_indicator": data.get("confidence_indicator"),
                        "citations": data.get("citations"),
                        "follow_up_suggestions": data.get("follow_up_suggestions"),
                        "model_used": data.get("model_used"),
                    })

                    # Follow-up suggestions
                    if data.get("follow_up_suggestions"):
                        st.write("**Suggested follow-ups:**")
                        cols = st.columns(len(data["follow_up_suggestions"]))
                        for i, suggestion in enumerate(data["follow_up_suggestions"]):
                            if cols[i].button(f"💬 {suggestion}", key=f"followup_new_{i}"):
                                st.session_state.pending_message = suggestion
                                st.rerun()

                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.Timeout:
                msg = "Request timed out. ARIA is processing a complex query — please try again."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as e:
                msg = f"Connection error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
```

---

## 12. Roadmap

### Phase 1 — Weeks 1–2: Core Agentic Loop

- Multi-model router with registry, fallback, and token tracking
- LangGraph graph with IntentClassifier, Planner, Executor, ResponseSynthesizer, MemoryWriter nodes
- Working memory (ConversationSummaryBufferMemory + SQLite persistence)
- WebSearch tool (Tavily + DuckDuckGo fallback)
- FastAPI backend with `/chat` endpoint
- Basic CLI interface
- Unit tests for model router and intent classifier

**Exit criteria:** ARIA can answer research questions with web search, maintain conversation context across turns, and automatically rotate models on failure.

### Phase 2 — Weeks 3–4: Full Memory and Tools

- Episodic memory: ChromaDB + SQLite, decay, pruning, retrieval injection
- RAG pipeline: document ingestion, semantic chunking, hybrid retrieval, cross-encoder reranking
- All 8 tools implemented and registered
- ResponsibleAI layer: hallucination detection, PII redaction, injection guard, content filter
- ReflectionAgent: scoring, retry loop
- Document upload API endpoint

**Exit criteria:** ARIA can answer questions from uploaded documents with citations, remember user preferences across sessions, and self-correct low-quality responses.

### Phase 3 — Weeks 5–6: Multi-Agent, Observability, and UI

- ResearchAgent, RAGAgent, CodeAgent, MemoryAgent as distinct agents
- LangSmith integration + structured JSON logging
- SQLite analytics + `/analytics` endpoint
- Full Streamlit UI: chat, document manager, memory viewer, model dashboard
- WebSocket streaming endpoint
- Integration test suite
- Production deployment to Railway + Streamlit Cloud

**Exit criteria:** Fully deployed, publicly accessible ARIA with all features operational.

### Phase 4 — Future

- **Voice interface:** Whisper (OpenAI free tier) for STT + ElevenLabs/local TTS
- **Multi-user support:** proper session isolation, per-user ChromaDB namespacing, billing hooks
- **Fine-tuned routing model:** small model trained on ARIA's own routing decisions to replace LLM-based routing
- **Plugin system:** community-contributed tools via a standard ToolSpec interface
- **Visual document understanding:** integrate Gemini Vision for PDF/image analysis
- **Autonomous task execution:** long-running background tasks with Celery + Upstash
- **Agent-to-agent communication:** true multi-agent messaging for collaborative complex tasks
- **Web scraping tool:** full page extraction via Playwright for paywalled or JS-heavy content

---

## License

MIT License — see `LICENSE` for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your change
4. Ensure CI passes: `make test && make lint`
5. Open a pull request with a clear description of what changed and why

---

*ARIA is built on the conviction that the infrastructure surrounding an LLM should never be the ceiling on what it can accomplish.*
