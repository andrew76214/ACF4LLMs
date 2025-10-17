# A2A Low Carbon System - Complete Code Tracing Documentation

**Last Updated**: 2025-10-16
**System Version**: 1.0
**Core Model**: Qwen3-4B-Thinking-2507 (fp8, 4.19 GB)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Execution Paths](#execution-paths)
6. [Configuration System](#configuration-system)
7. [Detailed Component Analysis](#detailed-component-analysis)
8. [Evaluation Scripts](#evaluation-scripts)
9. [Key Algorithms](#key-algorithms)
10. [File Reference](#file-reference)

---

## System Overview

The A2A (Agent-to-Agent) Low Carbon System is a multi-agent pipeline designed for question answering and mathematical reasoning with optimized VRAM usage. It leverages a small 4B parameter model (Qwen3-4B-Thinking) with intelligent agent coordination to achieve strong performance on multiple benchmarks.

### Key Features:
- **Multi-Agent Architecture**: Router → Search → Solver → Judge
- **Task-Adaptive Routing**: Automatically selects optimal strategy for math, QA, and multi-hop questions
- **Low VRAM Operation**: 4.19 GB model with fp8 quantization, ~13 GB total VRAM
- **Self-Consistency for Math**: 10 samples with majority voting
- **Hybrid Retrieval**: BM25 + Dense embeddings + Cross-encoder reranking
- **Math Verification**: Symbolic equivalence checking with Math-Verify
- **Carbon Tracking**: Built-in emissions monitoring with CodeCarbon

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│                       (Question/Query)                           │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ROUTER AGENT                                   │
│  • Dataset-based routing (GSM8K → math, HotpotQA → multihop)    │
│  • Content-based analysis (keywords, patterns, complexity)       │
│  • Outputs: RoutingDecision with task_type, use_search, params  │
└──────────────────┬──────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │  use_search?      │
         └──────┬──────┬─────┘
             YES│      │NO
                │      │
    ┌───────────▼      ▼────────────┐
    │   SEARCH AGENT          (skip)│
    │ • Tavily Web Search            │
    │ • BGE Embeddings               │
    │ • Hybrid Retrieval             │
    │ • BGE Reranker                 │
    └─────────┬──────────────────────┘
              │
              │ RetrievalContext
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      SOLVER AGENT                                 │
│  • Task-specific prompts (math/QA/multihop/multiple-choice)     │
│  • Model: Qwen3-4B-Thinking-2507 via vLLM                        │
│  • Strategies:                                                    │
│    - Math: Self-consistency (10 samples, temperature=0.6)        │
│    - QA/Multihop: Deterministic (1 sample, temperature=0.0)      │
│  • Outputs: SolverResponse with answer, reasoning, confidence    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ Predicted Answer
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                      JUDGE AGENT                                  │
│  • Math: Math-Verify symbolic equivalence                        │
│  • QA: F1 score and Exact Match                                  │
│  • Multihop: Answer + Supporting Facts metrics                   │
│  • Multiple Choice: Letter extraction and matching               │
│  • Outputs: JudgmentResult with correct/score/details            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     EVALUATION RESULTS                            │
│  • Accuracy, F1, EM metrics                                       │
│  • Per-question details (reasoning, time, confidence)            │
│  • Carbon emissions (optional)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### Core Agents

| Agent | File | Purpose | Input | Output |
|-------|------|---------|-------|--------|
| **RouterAgent** | `src/agents/router.py` | Task classification and routing | Question, dataset name, context | `RoutingDecision` |
| **SearchAgent** | `src/agents/search.py` | Web search and retrieval | Query | `RetrievalContext` with passages |
| **SolverAgent** | `src/agents/solver.py` | Answer generation | Question, routing, context | `SolverResponse` with answer |
| **JudgeAgent** | `src/agents/judge.py` | Answer evaluation | Prediction, gold answer, task type | `JudgmentResult` |

### Model Layer

| Component | File | Purpose |
|-----------|------|---------|
| **ModelManager** | `src/models/model_manager.py` | Direct vLLM model loading and management |
| **VLLMClient** | `src/models/vllm_client.py` | OpenAI-compatible API client for vLLM server |

### Orchestration

| Component | File | Purpose |
|-----------|------|---------|
| **Evaluator** | `src/evaluation/evaluator.py` | Orchestrates full pipeline for dataset evaluation |
| **A2APipeline** | `src/pipeline.py` | High-level interface for system interaction |

### Utilities

| Utility | File | Purpose |
|---------|------|---------|
| **Math-Verify** | `src/utils/math_verify.py` | Symbolic mathematical equivalence checking |
| **Metrics** | `src/utils/metrics.py` | F1, EM, majority voting, variance analysis |
| **Text Processing** | `src/utils/text_processing.py` | Answer extraction, normalization |
| **Data Processing** | `src/utils/data_processing.py` | Dataset loading and sampling |
| **Logging** | `src/utils/logging.py` | Structured logging with performance tracking |

---

## Data Flow

### Single Question Evaluation Flow

```python
# src/evaluation/evaluator.py:evaluate_single_question()

Question Input
    ↓
┌───────────────────────────────────────────────────────────────┐
│ STEP 1: ROUTING (RouterAgent)                                 │
│   routing_decision = router.route(question, dataset, context) │
│   • Determines: task_type, use_search, temperature, samples   │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 2: RETRIEVAL (SearchAgent) - If use_search == True       │
│   if routing_decision.use_search:                             │
│       if task_type == "multihop":                              │
│           context = search_agent.multi_hop_search(question)    │
│       else:                                                    │
│           context = search_agent.retrieve_and_rerank(question) │
│   • Tavily search → Hybrid ranking → Reranking                │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 3: SOLVING (SolverAgent)                                 │
│   solver_response = solver.solve(question, routing, context)   │
│                                                                │
│   3a. Build Prompt (task-specific templates)                  │
│       system_prompt, user_prompt = build_prompt()             │
│                                                                │
│   3b. Generate Answer                                          │
│       if self_consistency (math):                              │
│           # Generate 10 samples                                │
│           results = generate(num_samples=10, temp=0.6)         │
│           answer = majority_vote(results)                      │
│       else:                                                    │
│           # Single deterministic sample                        │
│           result = generate(num_samples=1, temp=0.0)           │
│           answer = result                                      │
│                                                                │
│   3c. Extract Final Answer                                     │
│       final_answer = extract_final_answer(raw_output)          │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 4: JUDGING (JudgeAgent)                                  │
│   judgment = judge.judge_single(prediction, gold, task_type)   │
│                                                                │
│   Based on task_type:                                          │
│   • math: Math-Verify symbolic equivalence                     │
│   • qa: F1 score and Exact Match                              │
│   • multihop: Answer EM/F1 + Supporting Facts EM/F1           │
│   • multiple_choice: Letter extraction and matching            │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
                Result Dict
        (correct, score, reasoning, timing)
```

---

## Execution Paths

### Path 1: Mathematical Question (GSM8K)

```python
# Example: "Janet has 5 apples and buys 3 more. How many does she have?"

# router.py:88-97
RoutingDecision(
    task_type="math",
    use_search=False,                    # Math doesn't need search
    prompt_style="math",
    decoding_strategy="self_consistency", # Key: Multiple samples
    num_samples=10,                       # Generate 10 answers
    temperature=0.6,                      # Higher temp for diversity
    reasoning="Dataset is mathematical"
)

# solver.py:358-456 (_solve_with_self_consistency)
1. Generate 10 samples with temperature=0.6
2. Extract answers from each sample using extract_final_answer()
3. Perform majority_vote() across all 10 answers
4. Return majority answer with agreement score as confidence

# judge.py:95-156 (_judge_math)
1. Parse prediction and gold using Math-Verify's parse_answer()
2. Check equivalence using verify(gold_parsed, pred_parsed)
3. Symbolic verification: "4" == "2+2" == "8/2" (all equivalent)
4. Return JudgmentResult(correct=True/False, score=1.0/0.0)
```

**File Path**: `router.py:88` → `solver.py:358` → `math_verify.py:79,135` → `judge.py:95`

### Path 2: Multi-hop QA Question (HotpotQA)

```python
# Example: "Were Scott Derrickson and Ed Wood of the same nationality?"

# router.py:111-121
RoutingDecision(
    task_type="multihop",
    use_search=True,                    # Needs external knowledge
    prompt_style="multihop_qa",
    decoding_strategy="deterministic",  # Single answer
    num_samples=1,
    temperature=0.0,                    # Deterministic
    reasoning="Dataset requires multi-hop reasoning"
)

# search.py:372-433 (multi_hop_search)
1. Hop 1: Direct Tavily search for main query
2. Extract entities from results (e.g., "Scott Derrickson", "Ed Wood")
3. Hop 2: Search with extracted entities
4. Combine and deduplicate results
5. Hybrid retrieval: BM25 (30%) + Dense (70%)
6. Cross-encoder reranking (BGE reranker)
7. Return top-k passages with relevance scores

# solver.py:221-227 (_build_prompt for multihop)
system_prompt = get_multihop_system_prompt()
user_prompt = f"Passages:\n{formatted_passages}\n\nQuestion: {question}"
# Emphasizes: Combine info from multiple sources, direct answer only

# solver.py:284-356 (_solve_single)
1. Generate single deterministic answer (temp=0.0)
2. Extract final answer (no JSON, just the answer text)

# judge.py:204-259 (_judge_multihop)
1. Parse prediction as JSON (answer + supporting_facts)
2. Compute answer EM and F1
3. Compute supporting facts EM and F1
4. Calculate joint metrics
5. Return JudgmentResult(correct=(joint_em==1), score=joint_f1)
```

**File Path**: `router.py:112` → `search.py:372` → `solver.py:221` → `judge.py:204`

### Path 3: Single-hop QA (SQuAD)

```python
# Example: "What is the capital of France?" with context paragraph

# router.py:123-135
RoutingDecision(
    task_type="qa",
    use_search=False,         # Context provided, no search needed
    prompt_style="qa",
    decoding_strategy="deterministic",
    num_samples=1,
    temperature=0.0,
    reasoning="Dataset is question-answering with context"
)

# solver.py:214-220 (_build_prompt for QA)
system_prompt = get_qa_system_prompt()
# "Answer questions accurately using the provided context"
user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

# solver.py:284-356 (_solve_single)
1. Generate answer from context (temp=0.0, 1 sample)
2. Extract final answer text

# judge.py:157-202 (_judge_qa)
1. Handle multiple gold answers (SQuAD has multiple valid answers)
2. Compute F1 against each gold answer
3. Take best F1 score
4. Correct if F1 > 0.8 OR exact match
5. Return JudgmentResult(correct=bool, score=best_f1)
```

**File Path**: `router.py:124` → `solver.py:214` → `judge.py:157`

---

## Configuration System

### Configuration Hierarchy

**File**: `src/configs/config.py`

```python
Config
├── ModelConfig (line 22-44)
│   ├── model_name: "unsloth/Qwen3-4B-Thinking-2507"
│   ├── quantization: "fp8"
│   ├── max_model_len: 32768
│   ├── gpu_memory_utilization: 0.5  # CRITICAL: Reduced for 13 GB VRAM
│   ├── temperature_math: 0.6
│   ├── temperature_qa: 0.0
│   ├── num_samples_math: 10         # Self-consistency samples
│   └── num_samples_qa: 1
│
├── SearchConfig (line 46-72)
│   ├── tavily_api_key: env.TAVILY_API_KEY
│   ├── max_search_results: 20
│   ├── search_depth: "advanced"
│   ├── embedding_model: "BAAI/bge-m3"
│   ├── reranker_model: "BAAI/bge-reranker-v2-m3"
│   ├── bm25_weight: 0.3
│   ├── dense_weight: 0.7
│   └── top_k_rerank: 5
│
├── RouterConfig (line 74-99)
│   ├── math_datasets: ["gsm8k", "math", "aime", "mathqa"]
│   ├── qa_datasets: ["squad", "nq", "triviaqa"]
│   ├── multihop_datasets: ["hotpotqa", "2wikimultihopqa"]
│   ├── multiple_choice_datasets: ["commonsenseqa", "arc", "hellaswag"]
│   └── [keyword lists for content-based routing]
│
├── EvaluationConfig (line 101-125)
│   ├── use_math_verify: True
│   ├── math_verify_timeout: 10
│   └── compute_detailed_metrics: True
│
├── DatasetConfig (line 127-165)
│   ├── dataset_paths: {dataset → path}
│   ├── sample_sizes: {dataset → sample_size}
│   └── answer_formats: {type → regex}
│
└── ExperimentConfig (line 167-189)
    ├── log_level: "INFO"
    ├── batch_size: 4
    ├── num_workers: 2
    └── random_seeds: [42, 123, 456, 789, 999]
```

### Key Configuration Changes

**gpu_memory_utilization**: Changed from `0.8` → `0.5` (line 28)
- **Reason**: RTX 4090 with 13.23 GB free VRAM
- **Impact**: Model uses ~6.6 GB instead of ~10.5 GB
- **Trade-off**: Slightly lower throughput, but fits in limited VRAM

---

## Detailed Component Analysis

### 1. RouterAgent (`src/agents/router.py`)

**Purpose**: Intelligently route questions to the optimal solving strategy

**Key Methods**:

#### `route(question, dataset, context)` → `RoutingDecision` (line 47)
Primary routing logic with two-stage decision:

```python
# Stage 1: Dataset-based routing (most reliable)
if dataset:
    if "gsm8k" in dataset → MATH strategy
    if "hotpotqa" in dataset → MULTIHOP strategy
    if "commonsenseqa" in dataset → MULTIPLE_CHOICE strategy
    if "squad" in dataset → QA strategy

# Stage 2: Content-based routing (fallback)
else:
    math_score = _compute_math_score(query)
    qa_score = _compute_qa_score(query)
    multihop_score = _compute_multihop_score(query)
    → Route to highest score
```

#### `_compute_math_score(query)` → float (line 198)
Mathematical content detection:
```python
score = 0.0
# Math keywords: solve, calculate, find, equation (+ 0.3 each, max 0.6)
# Math symbols: =, +, -, *, /, ^ (+0.4)
# Numbers in text (+0.1 per number, max 0.3)
# Operations: solve, find, calculate (+0.2)
# Contest indicators: AIME, olympiad (+0.3)
return min(score, 1.0)
```

#### `_compute_multihop_score(query)` → float (line 262)
Multi-hop reasoning detection:
```python
score = 0.0
# Conjunctions: and, but, also, both (+0.3 each, max 0.6)
# Multiple question words (≥2) (+0.4)
# Comparative/relational words: between, compared to (+0.3)
# Long queries (>15 words) (+0.2)
# Temporal/causal: before, after, because (+0.2)
return min(score, 1.0)
```

**Data Structures**:
```python
@dataclass
class RoutingDecision:
    task_type: str              # "math", "qa", "multihop", "multiple_choice"
    use_search: bool            # Whether to invoke SearchAgent
    prompt_style: str           # Template identifier for SolverAgent
    decoding_strategy: str      # "self_consistency" or "deterministic"
    num_samples: int            # Number of samples to generate
    temperature: float          # Sampling temperature
    reasoning: Optional[str]    # Explanation of routing decision
    confidence: float = 1.0     # Confidence in routing (0-1)
```

---

### 2. SearchAgent (`src/agents/search.py`)

**Purpose**: Web search, retrieval, and passage ranking

**Key Methods**:

#### `search(query, max_results)` → `List[SearchResult]` (line 127)
Tavily API web search:
```python
response = tavily_client.search(
    query=query,
    search_depth="advanced",  # Deep search mode
    max_results=20
)
# Returns: List of SearchResult(title, content, url, score, source)
```

#### `retrieve_and_rerank(query, passages, top_k)` → `RetrievalContext` (line 173)
Complete retrieval pipeline:
```python
1. Search (if passages not provided)
   passages = self.search(query, max_results=20)

2. Hybrid Retrieval
   ranked_passages = _hybrid_retrieval(query, passages, top_k * 2)
   # Combines BM25 (30%) and Dense (70%) scores

3. Cross-encoder Reranking
   reranked = _rerank_passages(query, ranked_passages, top_k)
   # Uses BGE reranker (BAAI/bge-reranker-v2-m3)

4. Return top-k passages
   return RetrievalContext(passages=reranked[:top_k], ...)
```

#### `_hybrid_retrieval(query, passages, top_k)` → `List[SearchResult]` (line 241)
Two-stage ranking:
```python
# Dense retrieval (BGE embeddings)
query_emb = embedding_model.encode_queries([query])
passage_embs = embedding_model.encode_corpus(passages)
dense_scores = cosine_similarity(query_emb, passage_embs)

# BM25 (TF-IDF approximation)
tfidf_matrix = vectorizer.fit_transform(passages + [query])
bm25_scores = cosine_similarity(query_vector, passage_vectors)

# Combine: 30% BM25 + 70% Dense
combined_scores = 0.3 * bm25_scores + 0.7 * dense_scores

# Rank by combined score
return sorted(passages, key=lambda x: x.score, reverse=True)[:top_k]
```

#### `multi_hop_search(query, max_hops=2)` → `RetrievalContext` (line 372)
Multi-hop search for complex queries:
```python
# Hop 1: Direct search
hop1_results = search(query, max_results=10)

# Hop 2+: Extract entities and search again
for hop in range(1, max_hops):
    # Extract named entities from recent results
    entities = extract_entities(hop1_results)
    additional_query = f"{entity} {query}"
    hop_results = search(additional_query, max_results=10)
    all_results.extend(hop_results)

# Deduplicate by URL
unique_results = deduplicate(all_results)

# Final reranking
return retrieve_and_rerank(query, unique_results, top_k=5)
```

**Data Structures**:
```python
@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    score: float        # Relevance score (0-1)
    source: str = "web"
    snippet_id: Optional[str] = None

@dataclass
class RetrievalContext:
    query: str
    passages: List[SearchResult]
    search_time: float
    total_results: int
    reranked: bool = False
```

---

### 3. SolverAgent (`src/agents/solver.py`)

**Purpose**: Generate answers using Qwen3-4B-Thinking model

**Key Methods**:

#### `solve(question, routing_decision, context)` → `SolverResponse` (line 142)
Main solving pipeline:
```python
1. Build Prompt
   system_prompt, user_prompt = _build_prompt(question, task_type, context)

2. Choose Strategy
   if routing_decision.decoding_strategy == "self_consistency":
       response = _solve_with_self_consistency(...)
   else:
       response = _solve_single(...)

3. Return SolverResponse
   return SolverResponse(
       answer=final_answer,
       reasoning=reasoning,
       confidence=confidence,
       raw_output=raw_output,
       metadata={...}
   )
```

#### `_build_prompt(question, task_type, context)` → (str, str) (line 193)
Task-specific prompt construction:

**Math Prompt** (line 32-50):
```python
system = """You are a careful mathematician. Think step by step and solve accurately.

Instructions:
- Show your reasoning in a <think> block
- Put your final numerical answer in \\boxed{} format
- For AIME problems, answers should be integers from 0 to 999
- Be precise with calculations and check your work

Example:
<think>
Let me solve this step by step...
[reasoning steps]
So the answer is 42.
</think>

\\boxed{42}"""

user = question  # Direct question, no context
```

**Multihop QA Prompt** (line 63-77):
```python
system = """You are an expert at multi-hop question answering. Combine information from multiple sources.

Instructions:
- Analyze all provided passages carefully
- Identify the key facts needed to answer the question
- Combine information from different passages when necessary
- Think step by step, but provide only the final answer

IMPORTANT: Provide ONLY the direct answer. Do not include explanations or JSON.

Examples:
- Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
→ Answer: "yes"
- Question: "What government position was held by the woman who portrayed Corliss Archer?"
→ Answer: "Chief of Protocol"
"""

user = f"Passages:\n{formatted_passages}\n\nQuestion: {question}"
```

#### `_solve_with_self_consistency(system, user, routing)` → `SolverResponse` (line 358)
Self-consistency decoding for math:
```python
# Generate multiple samples
results = model_client.generate(
    prompt=prompt,
    temperature=0.6,      # Higher temp for diversity
    num_samples=10        # Generate 10 different solutions
)

# Extract answers from all samples
answers = []
for result in results:
    raw_output = result.get("text")
    answer = extract_final_answer(raw_output, "math")
    answers.append(answer)

# Majority vote
from utils.metrics import majority_vote
majority_answer = majority_vote(answers)

# Calculate agreement (confidence)
agreement = answers.count(majority_answer) / len(answers)

return SolverResponse(
    answer=majority_answer,
    reasoning=results[0].get("reasoning"),
    confidence=agreement,  # Higher agreement = higher confidence
    metadata={
        "num_samples": 10,
        "all_answers": answers,
        "agreement": agreement
    }
)
```

**Example Self-Consistency**:
```
Question: "Janet has 5 apples and buys 3 more. How many does she have?"

Sample 1: \\boxed{8}
Sample 2: \\boxed{8}
Sample 3: \\boxed{8}
Sample 4: \\boxed{8}
Sample 5: \\boxed{8}
Sample 6: \\boxed{8}
Sample 7: \\boxed{8}
Sample 8: \\boxed{8}
Sample 9: \\boxed{8}
Sample 10: \\boxed{8}

Majority Vote: "8" (10/10 agreement = 100% confidence)
```

#### `_solve_single(system, user, routing)` → `SolverResponse` (line 284)
Single deterministic generation:
```python
response = model_client.generate_with_reasoning(
    prompt=user_prompt,
    system_message=system_prompt,
    temperature=0.0,   # Deterministic
    max_tokens=32768   # Full context
)

raw_output = response.get("content")
reasoning = response.get("reasoning")

# Extract final answer
final_answer = extract_final_answer(raw_output, task_type)

return SolverResponse(
    answer=final_answer,
    reasoning=reasoning,
    confidence=1.0,    # Single answer, full confidence
    raw_output=raw_output,
    metadata={"temperature": 0.0, "num_samples": 1}
)
```

**Prompt Templates Class** (`PromptTemplates`, line 29-106):
Centralized repository of all system prompts for different task types.

---

### 4. JudgeAgent (`src/agents/judge.py`)

**Purpose**: Evaluate predictions against gold answers

**Key Methods**:

#### `judge_single(prediction, gold_answer, task_type)` → `JudgmentResult` (line 52)
Route to appropriate evaluation method:
```python
if task_type == "math":
    return _judge_math(prediction, gold_answer, question_id)
elif task_type == "multihop":
    return _judge_multihop(prediction, gold_data, question_id)
elif task_type == "multiple_choice":
    return _judge_multiple_choice(prediction, gold_answer, question_id)
elif task_type in ["qa", "open_qa"]:
    return _judge_qa(prediction, gold_answer, question_id)
```

#### `_judge_math(prediction, gold_answer, question_id)` → `JudgmentResult` (line 95)
Math verification with symbolic equivalence:
```python
# Use Math-Verify for symbolic checking
from utils.math_verify import parse_answer, verify

try:
    # Parse both answers
    pred_parsed = parse_answer(prediction)    # "2+2" → Add(2, 2)
    gold_parsed = parse_answer(gold_answer)  # "4" → Integer(4)

    # Symbolic equivalence check
    is_correct = verify(gold_parsed, pred_parsed)
    # verify("4", "2+2") → True (symbolically equivalent)
    # verify("4", "3+3") → False

    return JudgmentResult(
        correct=is_correct,
        score=1.0 if is_correct else 0.0,
        details={
            "prediction_parsed": str(pred_parsed),
            "gold_parsed": str(gold_parsed),
            "method": "math_verify"
        }
    )
except:
    # Fallback to string matching
    pred_normalized = normalize_text(extract_boxed_answer(prediction))
    gold_normalized = normalize_text(gold_answer)
    is_correct = (pred_normalized == gold_normalized)
    return JudgmentResult(correct=is_correct, ...)
```

**Math-Verify Examples** (`utils/math_verify.py`):
```python
# Equivalent forms
verify("4", "4")           → True
verify("4", "2+2")         → True
verify("4", "8/2")         → True
verify("0.5", "1/2")       → True
verify("2", "sqrt(4)")     → True

# Non-equivalent
verify("4", "5")           → False
verify("4", "2+3")         → False

# LaTeX handling
verify("\\frac{1}{2}", "0.5")  → True
verify("\\sqrt{4}", "2")       → True
```

#### `_judge_qa(prediction, gold_answer, question_id)` → `JudgmentResult` (line 157)
QA evaluation with F1 and EM:
```python
# Handle multiple gold answers (SQuAD has multiple correct answers)
gold_answers = gold_answer if isinstance(gold_answer, list) else [gold_answer]

best_em = 0
best_f1 = 0.0

for gold in gold_answers:
    # Exact Match
    em = 1 if normalize_text(prediction) == normalize_text(gold) else 0
    best_em = max(best_em, em)

    # F1 Score (token overlap)
    f1 = compute_f1(prediction, gold)
    best_f1 = max(best_f1, f1)

# Question is correct if EM=1 OR F1 > 0.8
is_correct = (best_em == 1) or (best_f1 > 0.8)

return JudgmentResult(
    correct=is_correct,
    score=best_f1,  # Use F1 as continuous score
    details={"exact_match": best_em, "f1_score": best_f1}
)
```

#### `_judge_multihop(prediction, gold_data, question_id)` → `JudgmentResult` (line 204)
HotpotQA-style evaluation:
```python
# Parse prediction JSON
pred_data = extract_json_answer(prediction)  # {"answer": "...", "supporting_facts": [...]}

# Extract components
pred_answer = pred_data.get("answer")
pred_facts = pred_data.get("supporting_facts")
gold_answer = gold_data.get("answer")
gold_facts = gold_data.get("supporting_facts")

# Judge answer
answer_em = 1 if normalize_text(pred_answer) == normalize_text(gold_answer) else 0
answer_f1 = compute_f1(pred_answer, gold_answer)

# Judge supporting facts
support_metrics = compute_supporting_facts_metrics([pred_facts], [gold_facts])
support_em = support_metrics["supporting_facts_em"]
support_f1 = support_metrics["supporting_facts_f1"]

# Joint metrics (both must be correct)
joint_em = answer_em * support_em
joint_f1 = (answer_f1 + support_f1) / 2

return JudgmentResult(
    correct=(joint_em == 1),  # Strict: both answer and facts correct
    score=joint_f1,
    details={
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "support_em": support_em,
        "support_f1": support_f1,
        "joint_em": joint_em,
        "joint_f1": joint_f1
    }
)
```

#### `compute_dataset_metrics(judgment_results, task_type)` → Dict (line 341)
Aggregate metrics across all questions:
```python
# Basic metrics
correct_count = sum(1 for r in judgment_results if r.correct)
accuracy = correct_count / len(judgment_results)
avg_score = sum(r.score for r in judgment_results) / len(judgment_results)

metrics = {
    "accuracy": accuracy,
    "average_score": avg_score,
    "correct_count": correct_count,
    "total_count": len(judgment_results)
}

# Task-specific metrics
if task_type == "multihop":
    metrics.update({
        "answer_em": mean([r.details["answer_em"] for r in judgment_results]),
        "answer_f1": mean([r.details["answer_f1"] for r in judgment_results]),
        "support_em": mean([r.details["support_em"] for r in judgment_results]),
        "support_f1": mean([r.details["support_f1"] for r in judgment_results]),
        "joint_em": mean([r.details["joint_em"] for r in judgment_results]),
        "joint_f1": mean([r.details["joint_f1"] for r in judgment_results])
    })

return metrics
```

**Data Structures**:
```python
@dataclass
class JudgmentResult:
    correct: bool              # Binary correctness
    score: float               # Continuous score (F1, agreement, etc.)
    details: Dict[str, Any]    # Task-specific metrics
    explanation: Optional[str] = None  # Human-readable explanation
```

---

### 5. ModelManager (`src/models/model_manager.py`)

**Purpose**: Direct vLLM model loading and management

**Key Methods**:

#### `load_model()` → bool (line 40)
Load Qwen3-4B-Thinking with vLLM:
```python
model_kwargs = {
    "model": "unsloth/Qwen3-4B-Thinking-2507",
    "trust_remote_code": True,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.5,  # 50% of available VRAM
    "dtype": "auto",                 # fp16 or bf16 based on GPU
    "quantization": "fp8",           # fp8 quantization (4.19 GB)
    "enable_chunked_prefill": True,  # Memory efficient
    "max_num_batched_tokens": 8192   # Batch size limit
}

self.llm = LLM(**model_kwargs)
self.tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sanity check
test_result = self.generate("2+2", max_tokens=100, temperature=0.0)
assert "4" in test_result or "boxed{4}" in test_result
```

**Memory Profile**:
```
Model Size: 4.19 GB (fp8 quantized)
KV Cache: ~6 GB (at gpu_memory_utilization=0.5)
Overhead: ~2-3 GB (PagedAttention, CUDA context)
Total VRAM: ~13 GB (fits in RTX 4090 with 13.23 GB available)
```

#### `generate(prompt, max_tokens, temperature, num_samples)` → List[Dict] (line 160)
Generate responses with vLLM:
```python
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=0.95,
    max_tokens=max_tokens,
    n=num_samples,            # Number of samples
    stop=["</think>", "<|reasoning_end|>"],
    skip_special_tokens=False
)

outputs = self.llm.generate([prompt], sampling_params)

results = []
for output in outputs:
    for completion in output.outputs:
        # Extract reasoning if available
        reasoning, final_answer = _split_reasoning_and_answer(completion.text)
        results.append({
            "text": completion.text,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "finish_reason": completion.finish_reason
        })

return results
```

#### `_split_reasoning_and_answer(text)` → (str, str) (line 363)
Extract reasoning from <think> tags:
```python
# Look for <think>...</think> pattern
think_pattern = r'<think>(.*?)</think>'
think_match = re.search(think_pattern, text, re.DOTALL)

if think_match:
    reasoning = think_match.group(1).strip()
    final_answer = text[think_match.end():].strip()
    return reasoning, final_answer
else:
    return None, text.strip()
```

**Example Output**:
```
Input: "What is 5 + 3?"

Output:
{
    "text": "<think>\n5 + 3 = 8\nSo the answer is 8.\n</think>\n\n\\boxed{8}",
    "reasoning": "5 + 3 = 8\nSo the answer is 8.",
    "final_answer": "\\boxed{8}",
    "finish_reason": "stop"
}
```

---

### 6. Evaluator (`src/evaluation/evaluator.py`)

**Purpose**: Orchestrate full pipeline for dataset evaluation

**Key Methods**:

#### `evaluate_single_question(question, gold_answer, dataset, context)` → Dict (line 48)
Complete pipeline for one question:
```python
start_time = time.time()

# Step 1: Route
routing_decision = self.router.route(question, dataset, context)

# Step 2: Search (if needed)
retrieval_context = None
if routing_decision.use_search:
    if routing_decision.task_type == "multihop":
        retrieval_context = self.search_agent.multi_hop_search(question)
    else:
        retrieval_context = self.search_agent.retrieve_and_rerank(question)

# Step 3: Solve
solver_response = self.solver.solve(question, routing_decision, retrieval_context)

# Step 4: Judge
judgment = self.judge.judge_single(
    solver_response.answer,
    gold_answer,
    routing_decision.task_type
)

# Compile result
result = {
    "question": question,
    "prediction": solver_response.answer,
    "gold_answer": gold_answer,
    "correct": judgment.correct,
    "score": judgment.score,
    "task_type": routing_decision.task_type,
    "used_search": routing_decision.use_search,
    "processing_time": time.time() - start_time,
    "routing": {...},
    "solver": {...},
    "judgment": {...}
}

return result
```

#### `evaluate_dataset(dataset_name, sample_size, num_trials)` → Dict (line 143)
Full dataset evaluation with multiple trials:
```python
# Load dataset
data = load_dataset_by_name(dataset_name)
if sample_size:
    data = sample_dataset(data, sample_size)

# Run multiple trials for variance analysis
all_trial_results = []
trial_metrics = []

for trial in range(num_trials):
    # Evaluate all questions
    trial_results = []
    for i, item in enumerate(data):
        question, gold_answer = _extract_question_answer(item, dataset_name)
        result = self.evaluate_single_question(question, gold_answer, dataset_name)
        result["trial"] = trial
        result["question_id"] = i
        trial_results.append(result)

    # Compute trial metrics
    trial_judgments = [judge.judge_single(...) for r in trial_results]
    trial_metric = judge.compute_dataset_metrics(trial_judgments, dataset_name)
    trial_metrics.append(trial_metric["accuracy"])

    all_trial_results.extend(trial_results)

# Aggregate across trials
final_metrics = _aggregate_trial_metrics(all_trial_results, trial_metrics, dataset_name)

# Save results
save_results({
    "dataset": dataset_name,
    "metrics": final_metrics,
    "predictions": all_trial_results
}, f"{dataset_name}_results_{timestamp}.json")

return final_metrics
```

#### `_aggregate_trial_metrics(all_results, trial_accuracies, dataset_name)` → Dict (line 302)
Compute variance and aggregate metrics:
```python
# Basic metrics
accuracy = sum(r["correct"] for r in all_results) / len(all_results)

metrics = {
    "accuracy": accuracy,
    "correct_count": sum(r["correct"] for r in all_results),
    "total_count": len(all_results),
    "dataset": dataset_name
}

# Variance analysis (if multiple trials)
if len(trial_accuracies) > 1:
    metrics.update({
        "mean_accuracy": np.mean(trial_accuracies),
        "std_accuracy": np.std(trial_accuracies),
        "min_accuracy": np.min(trial_accuracies),
        "max_accuracy": np.max(trial_accuracies),
        "num_trials": len(trial_accuracies)
    })

# Task-specific detailed metrics
all_judgments = [judge.judge_single(...) for r in all_results]
detailed_metrics = judge.compute_dataset_metrics(all_judgments, task_type)
metrics.update(detailed_metrics)

return metrics
```

---

## Evaluation Scripts

### Main Evaluation Scripts

**Location**: `A2A_System_Package/`

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `run_gsm8k_evaluation_lazy.py` | GSM8K evaluation with lazy loading | On-demand model loading/unloading |
| `run_commonsenseqa_fast.py` | CommonsenseQA evaluation | Multiple choice questions |
| `run_hotpotqa_evaluation.py` | HotpotQA evaluation | Multi-hop QA with search |
| `run_complete_benchmark_evaluation.py` | All benchmarks | Comprehensive evaluation |

### GSM8K Lazy Loading Script

**File**: `run_gsm8k_evaluation_lazy.py`

**Purpose**: Memory-efficient GSM8K evaluation that loads/unloads model between batches

**Key Components**:

#### `LazyModelManager` (wrapper class)
```python
class LazyModelManager:
    """ModelManager wrapper that loads/unloads models on demand"""

    def __init__(self, model_config):
        self.model_config = model_config
        self.model_manager = None
        self.is_loaded = False

    def load(self):
        """Load the model into memory"""
        if not self.is_loaded:
            self.model_manager = ModelManager(self.model_config)
            success = self.model_manager.load_model()
            self.is_loaded = success
            return success
        return True

    def unload(self):
        """Unload the model from memory"""
        if self.is_loaded and self.model_manager:
            self.model_manager.cleanup()
            del self.model_manager
            self.model_manager = None
            self.is_loaded = False
            torch.cuda.empty_cache()  # Free VRAM
```

#### `GSM8KSolverAgent`
```python
class GSM8KSolverAgent(SolverAgent):
    """Enhanced SolverAgent with proven GSM8K prompt template"""

    def _get_gsm8k_system_prompt(self) -> str:
        return """You are a careful mathematician solving grade school math problems.

Instructions:
1. Read the problem carefully and identify what is being asked
2. Show your step-by-step reasoning inside <think></think> tags
3. After your reasoning, provide ONLY the final numerical answer in \\boxed{} format
4. The answer should be a single number (integer or decimal)
5. Do not include units or additional text after the boxed answer

Example:
<think>
Janet starts with 5 apples
She buys 3 more apples
Total = 5 + 3 = 8 apples
</think>

\\boxed{8}"""
```

#### Main Evaluation Loop
```python
def run_gsm8k_evaluation(dataset_name="gsm8k", limit=None):
    # Initialize lazy model manager
    lazy_model = LazyModelManager(config.model)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(limit))

    results = []

    try:
        # Load model
        lazy_model.load()

        # Initialize agents
        router = RouterAgent()
        solver = LazyGSM8KSolverAgent(lazy_model.model_manager)
        judge = JudgeAgent()

        # Evaluate each question
        for i, item in enumerate(dataset):
            question = item["question"]
            gold_answer = item["answer"].split("####")[1].strip()

            # Route (always math for GSM8K)
            routing = router.route(question, dataset="gsm8k")

            # Solve with self-consistency
            solver_response = solver.solve(question, routing)

            # Judge
            judgment = judge.judge_single(
                solver_response.answer,
                gold_answer,
                "math"
            )

            results.append({
                "question_id": i,
                "question": question,
                "prediction": solver_response.answer,
                "gold_answer": gold_answer,
                "correct": judgment.correct,
                "confidence": solver_response.confidence
            })

            # Optional: Unload between questions for extreme memory savings
            # lazy_model.unload()
            # lazy_model.load()  # Reload for next question

    finally:
        # Always unload model at end
        lazy_model.unload()

    # Compute metrics
    accuracy = sum(r["correct"] for r in results) / len(results)

    # Save results
    output_file = f"gsm8k_lazy_results_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)

    return {"accuracy": accuracy, "results": results}
```

**Workflow**:
```
1. Initialize LazyModelManager (model NOT loaded yet)
2. Load dataset
3. lazy_model.load() → Load model into VRAM
4. Process all questions with self-consistency
5. lazy_model.unload() → Free VRAM
6. Save results

Memory Timeline:
Before load():  ~6 GB VRAM (base CUDA)
After load():   ~13 GB VRAM (model + KV cache)
After unload(): ~6 GB VRAM (model freed)
```

---

## Key Algorithms

### 1. Self-Consistency Decoding

**Location**: `src/agents/solver.py:358-456`

**Purpose**: Improve math accuracy through majority voting

**Algorithm**:
```python
def self_consistency_solve(question, num_samples=10, temperature=0.6):
    """
    Self-consistency: Sample diverse solutions and take majority vote

    Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
               (Wang et al., ICLR 2023)
    """

    # Step 1: Generate multiple diverse samples
    samples = []
    for i in range(num_samples):
        # High temperature for diversity
        output = model.generate(
            prompt=question,
            temperature=temperature,  # 0.6 for diversity
            top_p=0.95
        )
        samples.append(output)

    # Step 2: Extract final answers from each sample
    answers = []
    for sample in samples:
        # Extract from \\boxed{...} or final line
        answer = extract_final_answer(sample)
        answers.append(answer)

    # Step 3: Majority voting
    # Count frequency of each unique answer
    answer_counts = Counter(answers)
    # Select most common answer
    majority_answer, count = answer_counts.most_common(1)[0]

    # Step 4: Calculate confidence (agreement rate)
    confidence = count / num_samples

    return {
        "answer": majority_answer,
        "confidence": confidence,
        "all_samples": samples,
        "all_answers": answers
    }

# Example:
# Question: "What is 15 + 27?"
#
# Sample 1: \\boxed{42}
# Sample 2: \\boxed{42}
# Sample 3: \\boxed{42}
# Sample 4: \\boxed{42}
# Sample 5: \\boxed{42}
# Sample 6: \\boxed{42}
# Sample 7: \\boxed{42}
# Sample 8: \\boxed{41}  ← Computation error
# Sample 9: \\boxed{42}
# Sample 10: \\boxed{42}
#
# Answer counts: {"42": 9, "41": 1}
# Majority: "42" with 90% agreement
# Final answer: "42" (confidence: 0.9)
```

**Why It Works**:
- Different reasoning paths may make different errors
- Correct answer is more likely to be consistent across samples
- Majority voting filters out random errors
- GSM8K improvement: +5-10% accuracy over greedy decoding

---

### 2. Hybrid Retrieval (BM25 + Dense)

**Location**: `src/agents/search.py:241-289`

**Purpose**: Combine lexical and semantic search

**Algorithm**:
```python
def hybrid_retrieval(query, passages, bm25_weight=0.3, dense_weight=0.7):
    """
    Hybrid retrieval combining BM25 (lexical) and Dense (semantic) scoring

    BM25: Good for exact keyword matches
    Dense: Good for semantic similarity
    Hybrid: Best of both worlds
    """

    # Step 1: Dense retrieval (semantic similarity)
    # Encode query and passages with BGE-M3
    query_embedding = embedding_model.encode_queries([query])      # [1, 1024]
    passage_embeddings = embedding_model.encode_corpus(passages)   # [N, 1024]

    # Compute cosine similarities
    dense_scores = cosine_similarity(query_embedding, passage_embeddings)[0]
    # dense_scores: [N] array of similarities (0-1)

    # Step 2: BM25 retrieval (lexical matching)
    # TF-IDF approximation of BM25
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(passages + [query])

    query_vector = tfidf_matrix[-1]        # Last row is query
    passage_vectors = tfidf_matrix[:-1]    # All other rows are passages

    # Compute similarities
    bm25_scores = cosine_similarity(query_vector, passage_vectors)[0]

    # Normalize to 0-1
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

    # Step 3: Combine scores (weighted sum)
    combined_scores = bm25_weight * bm25_scores + dense_weight * dense_scores
    # Default: 30% BM25 + 70% Dense

    # Step 4: Rank by combined score
    ranked_indices = np.argsort(combined_scores)[::-1]  # Descending order
    ranked_passages = [passages[i] for i in ranked_indices]

    return ranked_passages

# Example:
# Query: "What is the capital of France?"
#
# Passage 1: "Paris is the capital and largest city of France."
#   BM25: 0.9 (exact keywords: "Paris", "capital", "France")
#   Dense: 0.95 (high semantic similarity)
#   Combined: 0.3*0.9 + 0.7*0.95 = 0.935
#
# Passage 2: "The City of Light, also known as Paris, is famous for the Eiffel Tower."
#   BM25: 0.5 (has "Paris" but missing "capital", "France")
#   Dense: 0.85 (semantically related)
#   Combined: 0.3*0.5 + 0.7*0.85 = 0.745
#
# Passage 3: "France is a country in Western Europe with several beautiful cities."
#   BM25: 0.3 (has "France" but no "Paris" or "capital")
#   Dense: 0.6 (tangentially related)
#   Combined: 0.3*0.3 + 0.7*0.6 = 0.51
#
# Final ranking: Passage 1 > Passage 2 > Passage 3
```

---

### 3. Math-Verify Symbolic Equivalence

**Location**: `src/utils/math_verify.py:135-205`

**Purpose**: Check mathematical equivalence beyond string matching

**Algorithm**:
```python
def verify(gold_answer, predicted_answer, tolerance=1e-10):
    """
    Verify mathematical equivalence using SymPy

    Handles:
    - Different forms: "4" vs "2+2" vs "8/2"
    - Fractions: "1/2" vs "0.5"
    - LaTeX: "\\frac{1}{2}" vs "0.5"
    - Square roots: "sqrt(4)" vs "2"
    - Expressions: "x^2 + 2x + 1" vs "(x+1)^2"
    """

    # Step 1: Parse both answers into SymPy expressions
    try:
        gold_expr = sympify(normalize_latex(gold_answer), rational=True)
        pred_expr = sympify(normalize_latex(predicted_answer), rational=True)
    except:
        # Fallback to string comparison if parsing fails
        return str(gold_answer).strip() == str(predicted_answer).strip()

    # Step 2: Try direct equality
    if gold_expr == pred_expr:
        return True

    # Step 3: Try symbolic equivalence (.equals() method)
    if hasattr(gold_expr, 'equals') and hasattr(pred_expr, 'equals'):
        try:
            if gold_expr.equals(pred_expr):
                return True
        except:
            pass

    # Step 4: Try simplification and comparison
    try:
        diff = simplify(gold_expr - pred_expr)
        if diff == 0:
            return True
    except:
        pass

    # Step 5: Try numerical evaluation (last resort)
    try:
        gold_num = complex(N(gold_expr))
        pred_num = complex(N(pred_expr))

        if abs(gold_num - pred_num) < tolerance:
            return True
    except:
        pass

    # All methods failed - not equivalent
    return False

# Examples:
verify("4", "4")                    # True (direct equality)
verify("4", "2+2")                  # True (simplification: 2+2 → 4)
verify("4", "8/2")                  # True (simplification: 8/2 → 4)
verify("1/2", "0.5")                # True (numerical: 0.5 - 0.5 < 1e-10)
verify("sqrt(4)", "2")              # True (simplification: sqrt(4) → 2)
verify("\\frac{1}{2}", "0.5")       # True (LaTeX → 1/2 → 0.5)
verify("x^2+2x+1", "(x+1)^2")       # True (expand and simplify)
verify("pi", "3.14159")             # True (numerical within tolerance)

verify("4", "5")                    # False (4 ≠ 5)
verify("1/2", "1/3")                # False (0.5 ≠ 0.333...)
verify("x+1", "x+2")                # False ((x+1) - (x+2) = -1 ≠ 0)
```

**Why It's Critical for GSM8K**:
- Models may express answers in different forms
- "42" vs "42.0" vs "21*2" should all be correct
- Avoids false negatives from format differences
- Improves evaluation accuracy by 2-5%

---

## File Reference

### Complete File Structure

```
A2A_System_Package/
├── README.md                          # Main documentation
├── CODE_TRACING.md                    # This file
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── pipeline.py                    # Main A2APipeline class
│   ├── sequential_pipeline.py         # Alternative sequential pipeline
│   │
│   ├── agents/                        # Agent implementations
│   │   ├── __init__.py
│   │   ├── router.py                  # RouterAgent (task classification)
│   │   ├── solver.py                  # SolverAgent (answer generation)
│   │   ├── judge.py                   # JudgeAgent (answer evaluation)
│   │   └── search.py                  # SearchAgent (retrieval)
│   │
│   ├── models/                        # Model management
│   │   ├── __init__.py
│   │   ├── model_manager.py           # Direct vLLM model loading
│   │   └── vllm_client.py             # OpenAI-compatible client
│   │
│   ├── configs/                       # Configuration
│   │   ├── __init__.py
│   │   └── config.py                  # All configuration dataclasses
│   │
│   ├── evaluation/                    # Evaluation orchestration
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Main Evaluator class
│   │   └── sequential_evaluator.py    # Sequential evaluation
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── math_verify.py             # Math equivalence checking
│       ├── metrics.py                 # Evaluation metrics
│       ├── text_processing.py         # Text utilities
│       ├── data_processing.py         # Dataset loading
│       └── logging.py                 # Logging utilities
│
├── run_gsm8k_evaluation_lazy.py       # GSM8K evaluation (lazy loading)
├── run_commonsenseqa_fast.py          # CommonsenseQA evaluation
├── run_hotpotqa_evaluation.py         # HotpotQA evaluation
├── run_complete_benchmark_evaluation.py # All benchmarks
├── [... other evaluation scripts ...]
│
└── tests/                             # Test suite
    └── test_pipeline.py               # Pipeline tests
```

### Key File Locations by Functionality

#### Routing & Task Classification
- **Main**: `src/agents/router.py`
  - `RouterAgent.route()`: line 47
  - `_compute_math_score()`: line 198
  - `_compute_multihop_score()`: line 262

#### Search & Retrieval
- **Main**: `src/agents/search.py`
  - `SearchAgent.search()`: line 127 (Tavily search)
  - `retrieve_and_rerank()`: line 173 (full retrieval pipeline)
  - `_hybrid_retrieval()`: line 241 (BM25 + Dense)
  - `multi_hop_search()`: line 372 (multi-hop)

#### Answer Generation
- **Main**: `src/agents/solver.py`
  - `SolverAgent.solve()`: line 142
  - `_build_prompt()`: line 193 (task-specific prompts)
  - `_solve_with_self_consistency()`: line 358 (math)
  - `_solve_single()`: line 284 (QA)
- **Prompts**: `PromptTemplates` class (line 29-106)

#### Answer Evaluation
- **Main**: `src/agents/judge.py`
  - `JudgeAgent.judge_single()`: line 52
  - `_judge_math()`: line 95 (Math-Verify)
  - `_judge_qa()`: line 157 (F1 + EM)
  - `_judge_multihop()`: line 204 (HotpotQA)

#### Math Verification
- **Main**: `src/utils/math_verify.py`
  - `parse_answer()`: line 79 (string → SymPy)
  - `verify()`: line 135 (equivalence check)
  - `extract_boxed_answer()`: line 52 (\\boxed{} extraction)

#### Model Management
- **Main**: `src/models/model_manager.py`
  - `ModelManager.load_model()`: line 40
  - `generate()`: line 160 (vLLM generation)
  - `_split_reasoning_and_answer()`: line 363
- **Client**: `src/models/vllm_client.py`
  - `VLLMClient.generate_with_reasoning()`: line 101
  - `self_consistency_generate()`: line 190

#### Pipeline Orchestration
- **Main**: `src/evaluation/evaluator.py`
  - `Evaluator.evaluate_single_question()`: line 48
  - `evaluate_dataset()`: line 143
  - `_aggregate_trial_metrics()`: line 302
- **High-level**: `src/pipeline.py`
  - `A2APipeline.solve_question()`: line 102
  - `evaluate_dataset()`: line 148

#### Configuration
- **Main**: `src/configs/config.py`
  - `ModelConfig`: line 22 (gpu_memory_utilization=0.5 at line 28)
  - `SearchConfig`: line 46
  - `RouterConfig`: line 74
  - Global `config` instance: line 216

---

## Usage Examples

### Example 1: Solve a Single Math Question

```python
from src.pipeline import A2APipeline

# Initialize pipeline
pipeline = A2APipeline(use_vllm_server=False)
pipeline.initialize()

# Solve question
result = pipeline.solve_question(
    question="Janet has 5 apples and buys 3 more. How many does she have?",
    dataset="gsm8k"
)

print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Time: {result['processing_time']:.2f}s")

# Cleanup
pipeline.cleanup()
```

### Example 2: Evaluate on GSM8K Dataset

```python
from src.evaluation.evaluator import Evaluator
from src.models.model_manager import ModelManager

# Initialize model
model_manager = ModelManager()
model_manager.load_model()

# Initialize evaluator
evaluator = Evaluator(model_manager)

# Evaluate
results = evaluator.evaluate_dataset(
    dataset_name="gsm8k",
    sample_size=100,  # First 100 questions
    num_trials=1      # Single run
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Correct: {results['correct_count']}/{results['total_count']}")

# Cleanup
model_manager.cleanup()
```

### Example 3: Multi-hop Question with Search

```python
from src.agents.router import RouterAgent
from src.agents.search import SearchAgent
from src.agents.solver import SolverAgent
from src.agents.judge import JudgeAgent
from src.models.model_manager import ModelManager

# Initialize
model_manager = ModelManager()
model_manager.load_model()

router = RouterAgent()
search_agent = SearchAgent()
solver = SolverAgent(model_manager)
judge = JudgeAgent()

# Question
question = "Were Scott Derrickson and Ed Wood of the same nationality?"
gold_answer = "yes"

# Step 1: Route
routing = router.route(question, dataset="hotpotqa")
print(f"Routed to: {routing.task_type}")
print(f"Use search: {routing.use_search}")

# Step 2: Search
if routing.use_search:
    context = search_agent.multi_hop_search(question)
    print(f"Retrieved {len(context.passages)} passages")

# Step 3: Solve
response = solver.solve(question, routing, context)
print(f"Predicted: {response.answer}")

# Step 4: Judge
judgment = judge.judge_single(response.answer, gold_answer, "multihop")
print(f"Correct: {judgment.correct}")

# Cleanup
model_manager.cleanup()
```

---

## Performance Characteristics

### VRAM Usage
- **Model Size**: 4.19 GB (fp8 quantized)
- **KV Cache**: ~6-7 GB (at gpu_memory_utilization=0.5)
- **Overhead**: ~2-3 GB (CUDA, PagedAttention)
- **Total**: ~13 GB (fits RTX 4090 with 13.23 GB available)

### Throughput
- **Math (self-consistency)**: ~2-3 questions/minute (10 samples each)
- **QA (deterministic)**: ~10-15 questions/minute (1 sample each)
- **With search**: Add ~5-10s per question

### Accuracy (Approximate)
- **GSM8K**: ~65-75% (with self-consistency)
- **CommonsenseQA**: ~70-80% (multiple choice)
- **HotpotQA**: ~50-60% (multi-hop with search)

---

## Carbon Tracking

The system includes optional carbon emissions tracking via CodeCarbon:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# Run evaluation
results = evaluator.evaluate_dataset("gsm8k")

emissions = tracker.stop()
print(f"Carbon emissions: {emissions:.6f} kg CO2")
```

**Typical Emissions**:
- GSM8K (1319 questions): ~0.01-0.02 kg CO2
- Full benchmark suite: ~0.05-0.1 kg CO2

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce gpu_memory_utilization in config.py
gpu_memory_utilization: float = 0.4  # Down from 0.5

# Or use lazy loading
from run_gsm8k_evaluation_lazy import LazyModelManager
lazy_model = LazyModelManager(config.model)
```

### Search API Errors
```python
# Check Tavily API key
echo $TAVILY_API_KEY

# Or set in code
import os
os.environ["TAVILY_API_KEY"] = "your-key-here"
```

### Math-Verify Errors
```python
# Math-Verify depends on SymPy
pip install sympy

# Test Math-Verify
python -c "from src.utils.math_verify import test_math_verify; test_math_verify()"
```

---

## Development Notes

### Adding New Datasets
1. Add dataset config in `config.py:DatasetConfig`
2. Add dataset name to appropriate list in `RouterConfig`
3. Implement loading in `data_processing.py:load_dataset_by_name()`
4. Add answer extraction in `evaluator.py:_extract_question_answer()`

### Adding New Task Types
1. Add routing logic in `router.py:_route_by_dataset()` or `_route_by_content()`
2. Add prompt template in `solver.py:PromptTemplates`
3. Add judgment logic in `judge.py:_judge_<task_type>()`
4. Update `RoutingDecision.task_type` type hints

### Modifying Prompts
All prompts are centralized in `src/agents/solver.py:PromptTemplates` (line 29-106). Modify there to affect all evaluations.

---

## Extending the System

This section provides detailed instructions for extending the A2A system with new agents, features, and optimization techniques.

### Adding New Agents

The A2A system is designed to be extensible. You can add new agents to the pipeline for specialized tasks.

#### Step 1: Create New Agent Class

Create a new file in `src/agents/` following the agent interface pattern:

```python
# src/agents/your_new_agent.py

"""
YourNewAgent for A2A Pipeline
Description of what this agent does
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger("your_new_agent")

@dataclass
class YourAgentResponse:
    """Response from your agent"""
    result: Any
    confidence: float
    metadata: Dict[str, Any]

class YourNewAgent:
    """
    Your new agent that performs a specific task

    Purpose: Explain the purpose of this agent
    Input: Describe expected inputs
    Output: Describe output format
    """

    def __init__(self, config=None):
        """Initialize your agent"""
        self.config = config
        logger.info("YourNewAgent initialized")

    def process(self, input_data: Any, **kwargs) -> YourAgentResponse:
        """
        Main processing method

        Args:
            input_data: Input to process
            **kwargs: Additional parameters

        Returns:
            YourAgentResponse with results
        """
        try:
            # Your agent logic here
            result = self._perform_task(input_data)

            return YourAgentResponse(
                result=result,
                confidence=1.0,
                metadata={"processing_time": 0.0}
            )
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            raise

    def _perform_task(self, input_data: Any) -> Any:
        """Internal method to perform the main task"""
        # Implement your agent's core logic
        pass
```

#### Step 2: Integrate Agent into Pipeline

**Option A: Add to Evaluator** (`src/evaluation/evaluator.py`):

```python
# In Evaluator.__init__()
from agents.your_new_agent import YourNewAgent

def __init__(self, model_client: Optional[Any] = None):
    # ... existing code ...
    self.your_agent = YourNewAgent()
    logger.info("Added YourNewAgent to pipeline")

# In evaluate_single_question()
def evaluate_single_question(self, question, gold_answer, dataset, context):
    # ... existing routing and search ...

    # Step 2.5: Apply your agent (if needed)
    if some_condition:
        agent_result = self.your_agent.process(question)
        # Use agent_result in subsequent steps

    # ... continue with solver and judge ...
```

**Option B: Create Parallel Agent Pipeline**:

```python
# src/parallel_pipeline.py

from agents.router import RouterAgent
from agents.your_new_agent import YourNewAgent
from agents.solver import SolverAgent

class ParallelAgentPipeline:
    """Pipeline with parallel agent execution"""

    def __init__(self):
        self.router = RouterAgent()
        self.agent1 = YourNewAgent()
        self.solver = SolverAgent()

    def process_parallel(self, question: str):
        """Process question through multiple agents in parallel"""
        import concurrent.futures

        routing = self.router.route(question)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit parallel tasks
            future1 = executor.submit(self.agent1.process, question)
            future2 = executor.submit(self.solver.solve, question, routing)

            # Gather results
            result1 = future1.result()
            result2 = future2.result()

        # Combine results
        return self._merge_results(result1, result2)
```

#### Step 3: Add Agent to Configuration

```python
# src/configs/config.py

@dataclass
class YourAgentConfig:
    """Configuration for your new agent"""
    enable: bool = True
    parameter1: float = 0.5
    parameter2: int = 10

# In Config class
@dataclass
class Config:
    # ... existing configs ...
    your_agent: YourAgentConfig = None

    def __post_init__(self):
        # ... existing code ...
        if self.your_agent is None:
            self.your_agent = YourAgentConfig()
```

---

### Adding Advanced Router Types

You can implement alternative routing strategies beyond the current dataset+content routing.

#### Example: LLM-Based Meta-Router

```python
# src/agents/meta_router.py

"""
Meta-Router using LLM to classify questions
More flexible than rule-based routing
"""

from agents.router import RouterAgent, RoutingDecision
from models.vllm_client import VLLMClient

class MetaRouter(RouterAgent):
    """
    Meta-router that uses an LLM to classify tasks
    More flexible and adaptive than rule-based routing
    """

    def __init__(self, model_client: VLLMClient):
        super().__init__()
        self.model_client = model_client

    def route(self, query: str, dataset: Optional[str] = None,
              context: Optional[str] = None) -> RoutingDecision:
        """
        Route using LLM classification

        Falls back to parent class routing if LLM classification fails
        """

        # Try LLM classification first
        try:
            task_type = self._llm_classify(query)

            # Map task type to routing parameters
            if task_type == "math":
                return RoutingDecision(
                    task_type="math",
                    use_search=False,
                    prompt_style="math",
                    decoding_strategy="self_consistency",
                    num_samples=10,
                    temperature=0.6,
                    reasoning="LLM classified as math"
                )
            elif task_type == "multihop":
                return RoutingDecision(
                    task_type="multihop",
                    use_search=True,
                    prompt_style="multihop_qa",
                    decoding_strategy="deterministic",
                    num_samples=1,
                    temperature=0.0,
                    reasoning="LLM classified as multihop"
                )
            # ... more task types ...

        except Exception as e:
            logger.warning(f"LLM routing failed, using rule-based: {e}")
            # Fallback to parent class rule-based routing
            return super().route(query, dataset, context)

    def _llm_classify(self, query: str) -> str:
        """Use LLM to classify query type"""

        classification_prompt = f"""Classify the following question into ONE category:
- math: Mathematical problems requiring calculations
- qa: Simple factual questions
- multihop: Questions requiring multiple pieces of information
- multiple_choice: Multiple choice questions

Question: {query}

Category (respond with only one word):"""

        response = self.model_client.generate_with_reasoning(
            prompt=classification_prompt,
            temperature=0.0,
            max_tokens=20
        )

        # Extract classification from response
        classification = response.get("content", "").strip().lower()

        # Validate classification
        valid_types = ["math", "qa", "multihop", "multiple_choice"]
        if classification in valid_types:
            return classification
        else:
            raise ValueError(f"Invalid classification: {classification}")
```

#### Example: Ensemble Router

```python
# src/agents/ensemble_router.py

"""
Ensemble Router combining multiple routing strategies
"""

from agents.router import RouterAgent, RoutingDecision
from agents.meta_router import MetaRouter
from typing import List

class EnsembleRouter:
    """
    Combines multiple routers using voting or confidence weighting
    """

    def __init__(self, routers: List[RouterAgent], strategy="voting"):
        """
        Args:
            routers: List of router instances
            strategy: "voting" or "confidence_weighted"
        """
        self.routers = routers
        self.strategy = strategy

    def route(self, query: str, dataset: Optional[str] = None,
              context: Optional[str] = None) -> RoutingDecision:
        """Route using ensemble of routers"""

        # Get decisions from all routers
        decisions = []
        for router in self.routers:
            try:
                decision = router.route(query, dataset, context)
                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Router {type(router).__name__} failed: {e}")

        if not decisions:
            raise RuntimeError("All routers failed")

        # Combine decisions based on strategy
        if self.strategy == "voting":
            return self._majority_vote(decisions)
        elif self.strategy == "confidence_weighted":
            return self._weighted_decision(decisions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _majority_vote(self, decisions: List[RoutingDecision]) -> RoutingDecision:
        """Select most common routing decision"""
        from collections import Counter

        # Count task types
        task_types = [d.task_type for d in decisions]
        most_common_type = Counter(task_types).most_common(1)[0][0]

        # Return first decision with most common type
        for decision in decisions:
            if decision.task_type == most_common_type:
                decision.reasoning = f"Ensemble vote: {most_common_type}"
                return decision

    def _weighted_decision(self, decisions: List[RoutingDecision]) -> RoutingDecision:
        """Weight by confidence scores"""
        # Find decision with highest confidence
        best_decision = max(decisions, key=lambda d: d.confidence)
        best_decision.reasoning = f"Highest confidence: {best_decision.confidence:.2f}"
        return best_decision
```

---

### Adding Model Quantization

Implement additional quantization methods to further reduce VRAM usage.

#### Step 1: Add Quantization Configuration

```python
# src/configs/config.py

@dataclass
class ModelConfig:
    # ... existing fields ...

    # Quantization options
    quantization: str = "fp8"  # "fp8", "int8", "int4", "awq", "gptq", "none"
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        # Set quantization-specific configs
        if self.quantization == "int4" and self.quantization_config is None:
            self.quantization_config = {
                "bits": 4,
                "group_size": 128,
                "desc_act": True
            }
        elif self.quantization == "awq" and self.quantization_config is None:
            self.quantization_config = {
                "zero_point": True,
                "q_group_size": 128
            }
```

#### Step 2: Implement Quantization Loader

```python
# src/models/quantized_model_manager.py

"""
Model Manager with advanced quantization support
"""

from models.model_manager import ModelManager
import torch

class QuantizedModelManager(ModelManager):
    """
    Extended ModelManager with support for various quantization methods
    """

    def load_model(self) -> bool:
        """Load model with specified quantization"""

        quantization = self.config.quantization.lower()

        if quantization == "int8":
            return self._load_int8_model()
        elif quantization == "int4":
            return self._load_int4_model()
        elif quantization == "awq":
            return self._load_awq_model()
        elif quantization == "gptq":
            return self._load_gptq_model()
        else:
            # Use parent class for fp8 or none
            return super().load_model()

    def _load_int8_model(self) -> bool:
        """Load model with INT8 quantization"""
        try:
            from transformers import BitsAndBytesConfig

            logger.info("Loading model with INT8 quantization")

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

            # Load with vLLM (if supported) or transformers
            model_kwargs = {
                "model": self.config.model_name,
                "quantization": "bitsandbytes",
                "quantization_config": quantization_config,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len
            }

            self.llm = LLM(**model_kwargs)
            self.is_loaded = True

            logger.info(f"INT8 model loaded successfully")
            logger.info(f"Estimated VRAM usage: ~{self._estimate_vram(8):.2f} GB")

            return True

        except Exception as e:
            logger.error(f"Failed to load INT8 model: {e}")
            return False

    def _load_int4_model(self) -> bool:
        """Load model with INT4 quantization (GPTQ or AWQ)"""
        try:
            from transformers import BitsAndBytesConfig

            logger.info("Loading model with INT4 quantization")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"  # NormalFloat4
            )

            model_kwargs = {
                "model": self.config.model_name,
                "quantization": "bitsandbytes",
                "quantization_config": quantization_config,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len
            }

            self.llm = LLM(**model_kwargs)
            self.is_loaded = True

            logger.info(f"INT4 model loaded successfully")
            logger.info(f"Estimated VRAM usage: ~{self._estimate_vram(4):.2f} GB")

            return True

        except Exception as e:
            logger.error(f"Failed to load INT4 model: {e}")
            return False

    def _load_awq_model(self) -> bool:
        """Load AWQ quantized model"""
        try:
            logger.info("Loading AWQ quantized model")

            # AWQ models usually have '-AWQ' in the model name
            awq_model_name = self.config.model_name
            if "-AWQ" not in awq_model_name:
                awq_model_name = f"{awq_model_name}-AWQ"
                logger.info(f"Trying AWQ model: {awq_model_name}")

            model_kwargs = {
                "model": awq_model_name,
                "quantization": "awq",
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "trust_remote_code": True
            }

            self.llm = LLM(**model_kwargs)
            self.is_loaded = True

            logger.info("AWQ model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load AWQ model: {e}")
            return False

    def _load_gptq_model(self) -> bool:
        """Load GPTQ quantized model"""
        try:
            logger.info("Loading GPTQ quantized model")

            # GPTQ models usually have '-GPTQ' in the model name
            gptq_model_name = self.config.model_name
            if "-GPTQ" not in gptq_model_name:
                gptq_model_name = f"{gptq_model_name}-GPTQ"
                logger.info(f"Trying GPTQ model: {gptq_model_name}")

            model_kwargs = {
                "model": gptq_model_name,
                "quantization": "gptq",
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "trust_remote_code": True
            }

            self.llm = LLM(**model_kwargs)
            self.is_loaded = True

            logger.info("GPTQ model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load GPTQ model: {e}")
            return False

    def _estimate_vram(self, bits: int) -> float:
        """
        Estimate VRAM usage based on quantization

        Args:
            bits: Quantization bits (4, 8, 16)

        Returns:
            Estimated VRAM in GB
        """
        # Qwen3-4B has approximately 4B parameters
        num_params = 4e9

        # Calculate model size in bytes
        bytes_per_param = bits / 8
        model_size_gb = (num_params * bytes_per_param) / (1024**3)

        # Add KV cache overhead (~1.5x model size at gpu_memory_utilization=0.5)
        kv_cache_gb = model_size_gb * 1.5

        # Add CUDA overhead (~2 GB)
        cuda_overhead_gb = 2.0

        total_vram_gb = model_size_gb + kv_cache_gb + cuda_overhead_gb

        return total_vram_gb
```

#### Step 3: Use Quantized Model

```python
# In your evaluation script
from src.models.quantized_model_manager import QuantizedModelManager
from src.configs.config import config

# Set quantization in config
config.model.quantization = "int4"  # or "int8", "awq", "gptq"

# Load quantized model
model_manager = QuantizedModelManager()
success = model_manager.load_model()

if success:
    print(f"Model loaded with {config.model.quantization} quantization")
    print(f"VRAM usage: ~{model_manager._estimate_vram(4):.2f} GB")
```

**Quantization Comparison**:

| Method | Bits | Model Size | Total VRAM | Speed | Accuracy Loss |
|--------|------|------------|------------|-------|---------------|
| **FP16** | 16 | ~8 GB | ~20 GB | 1.0x | 0% |
| **FP8** | 8 | ~4 GB | ~13 GB | 1.2x | <1% |
| **INT8** | 8 | ~4 GB | ~12 GB | 1.5x | 1-2% |
| **INT4** | 4 | ~2 GB | ~8 GB | 2.0x | 2-5% |
| **AWQ** | 4 | ~2 GB | ~7 GB | 2.5x | 1-3% |
| **GPTQ** | 4 | ~2 GB | ~7 GB | 2.5x | 1-3% |

---

### Adding Model Pruning

Implement model pruning to reduce model size while maintaining performance.

#### Step 1: Create Pruning Utility

```python
# src/models/pruning.py

"""
Model Pruning for A2A System
Reduces model size by removing less important weights
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from utils.logging import get_logger

logger = get_logger("pruning")

class ModelPruner:
    """
    Prune model weights to reduce size

    Supports:
    - Magnitude-based pruning
    - Structured pruning (neurons, channels)
    - Unstructured pruning (individual weights)
    """

    def __init__(self, pruning_ratio: float = 0.3, method: str = "magnitude"):
        """
        Args:
            pruning_ratio: Fraction of weights to prune (0.0-1.0)
            method: "magnitude", "structured", or "unstructured"
        """
        self.pruning_ratio = pruning_ratio
        self.method = method
        logger.info(f"Initialized pruner: {method}, ratio={pruning_ratio}")

    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune model weights

        Args:
            model: PyTorch model to prune

        Returns:
            Pruned model
        """
        if self.method == "magnitude":
            return self._magnitude_pruning(model)
        elif self.method == "structured":
            return self._structured_pruning(model)
        elif self.method == "unstructured":
            return self._unstructured_pruning(model)
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")

    def _magnitude_pruning(self, model: nn.Module) -> nn.Module:
        """
        Prune weights with smallest magnitude

        This is the simplest and most effective pruning method
        """
        logger.info("Applying magnitude-based pruning...")

        import torch.nn.utils.prune as prune

        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_ratio,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        logger.info(f"Pruned {self.pruning_ratio*100:.1f}% of weights")

        # Calculate actual sparsity
        sparsity = self._calculate_sparsity(model)
        logger.info(f"Actual sparsity: {sparsity*100:.1f}%")

        return model

    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """
        Prune entire neurons/channels

        More aggressive but can be faster in practice
        """
        logger.info("Applying structured pruning...")

        import torch.nn.utils.prune as prune

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune neurons in linear layers
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.pruning_ratio,
                    n=2,
                    dim=0  # Prune output neurons
                )
                prune.remove(module, 'weight')

        logger.info(f"Pruned {self.pruning_ratio*100:.1f}% of neurons")
        return model

    def _unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """
        Prune individual weights

        Most flexible but requires sparse computation support
        """
        logger.info("Applying unstructured pruning...")

        import torch.nn.utils.prune as prune

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=self.pruning_ratio
                )
                prune.remove(module, 'weight')

        logger.info(f"Pruned {self.pruning_ratio*100:.1f}% of weights")
        return model

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate actual sparsity of model"""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params
        return sparsity

    def fine_tune_pruned_model(self, model: nn.Module, train_dataset,
                                num_epochs: int = 3) -> nn.Module:
        """
        Fine-tune pruned model to recover performance

        Args:
            model: Pruned model
            train_dataset: Training data
            num_epochs: Number of fine-tuning epochs

        Returns:
            Fine-tuned model
        """
        logger.info(f"Fine-tuning pruned model for {num_epochs} epochs...")

        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataset:
                optimizer.zero_grad()

                outputs = model(batch['input_ids'])
                loss = criterion(outputs, batch['labels'])

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataset)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        logger.info("Fine-tuning complete")
        return model

class PrunedModelManager:
    """
    Model Manager that loads and applies pruning
    """

    def __init__(self, model_config, pruning_ratio: float = 0.3):
        from models.model_manager import ModelManager

        self.base_manager = ModelManager(model_config)
        self.pruner = ModelPruner(pruning_ratio=pruning_ratio, method="magnitude")

    def load_and_prune_model(self) -> bool:
        """Load model and apply pruning"""

        # Load base model
        success = self.base_manager.load_model()
        if not success:
            return False

        # Get underlying model (vLLM exposes model)
        model = self.base_manager.llm.model

        # Apply pruning
        pruned_model = self.pruner.prune_model(model)

        # Replace model
        self.base_manager.llm.model = pruned_model

        logger.info("Model loaded and pruned successfully")
        return True

    def generate(self, *args, **kwargs):
        """Forward generate to base manager"""
        return self.base_manager.generate(*args, **kwargs)
```

#### Step 2: Use Pruned Model

```python
# Example: Load and prune model
from src.models.pruning import PrunedModelManager
from src.configs.config import config

# Create pruned model manager
pruned_manager = PrunedModelManager(
    model_config=config.model,
    pruning_ratio=0.3  # Prune 30% of weights
)

# Load and prune
success = pruned_manager.load_and_prune_model()

if success:
    print("Model loaded and pruned")

    # Use like normal model manager
    results = pruned_manager.generate(
        prompt="What is 2+2?",
        temperature=0.0
    )
    print(f"Result: {results[0]['text']}")
```

**Pruning Benefits**:
- **VRAM Reduction**: 30% pruning → ~20-30% VRAM savings
- **Speed Improvement**: Sparse operations can be 1.5-2x faster
- **Accuracy Impact**: Usually <3% with fine-tuning, <5% without
- **Best for**: Large models where VRAM is critical

---

### Performance Optimization Tips

#### 1. Batch Processing

```python
# src/utils/batch_processor.py

"""
Efficient batch processing for multiple questions
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 8
    max_concurrent: int = 4
    timeout: int = 60

class BatchProcessor:
    """Process multiple questions efficiently"""

    def __init__(self, evaluator, config: BatchConfig = None):
        self.evaluator = evaluator
        self.config = config or BatchConfig()

    def process_batch(self, questions: List[str],
                     gold_answers: List[Any],
                     dataset: str) -> List[Dict]:
        """
        Process questions in optimized batches

        Reduces overhead from repeated model loading/inference setup
        """
        results = []

        for i in range(0, len(questions), self.config.batch_size):
            batch_questions = questions[i:i + self.config.batch_size]
            batch_answers = gold_answers[i:i + self.config.batch_size]

            # Process batch
            batch_results = self._process_single_batch(
                batch_questions, batch_answers, dataset
            )
            results.extend(batch_results)

        return results

    def _process_single_batch(self, questions, answers, dataset):
        """Process a single batch"""
        # Implement batched inference here
        # This is more efficient than one-by-one processing
        pass
```

#### 2. Caching Results

```python
# src/utils/cache.py

"""
Cache for storing intermediate results
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional

class ResultCache:
    """Cache for expensive operations"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        cache_file = self.cache_dir / f"{self._hash(key)}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def set(self, key: str, value: Any):
        """Cache result"""
        cache_file = self.cache_dir / f"{self._hash(key)}.json"

        with open(cache_file, 'w') as f:
            json.dump(value, f)

    def _hash(self, key: str) -> str:
        """Generate hash for key"""
        return hashlib.md5(key.encode()).hexdigest()

# Usage:
cache = ResultCache()

# Check cache before expensive operation
question = "What is 2+2?"
cached_result = cache.get(question)

if cached_result:
    result = cached_result
else:
    result = solver.solve(question)
    cache.set(question, result)
```

---

### Complete Example: Custom Agent + Quantization + Pruning

```python
# example_custom_system.py

"""
Example: Build custom A2A system with:
- New verification agent
- INT4 quantization
- 30% model pruning
"""

from src.agents.router import RouterAgent
from src.agents.solver import SolverAgent
from src.agents.judge import JudgeAgent
from src.models.quantized_model_manager import QuantizedModelManager
from src.models.pruning import ModelPruner
from src.configs.config import config

# Custom verification agent
class VerificationAgent:
    """Agent that verifies answers with multiple strategies"""

    def verify(self, answer: str, gold: str, task_type: str) -> Dict:
        """Multi-strategy verification"""

        results = {
            "exact_match": answer.strip() == gold.strip(),
            "case_insensitive": answer.lower() == gold.lower(),
            "numeric": self._check_numeric(answer, gold),
            "semantic": self._check_semantic(answer, gold)
        }

        # Overall verification
        verified = any(results.values())
        confidence = sum(results.values()) / len(results)

        return {
            "verified": verified,
            "confidence": confidence,
            "strategies": results
        }

    def _check_numeric(self, answer: str, gold: str) -> bool:
        """Check if numerical values match"""
        try:
            return float(answer) == float(gold)
        except:
            return False

    def _check_semantic(self, answer: str, gold: str) -> bool:
        """Check semantic similarity (simplified)"""
        # Use sentence transformers or similar
        return False  # Placeholder

# Configure quantization
config.model.quantization = "int4"
config.model.gpu_memory_utilization = 0.4

# Load quantized model
print("Loading INT4 quantized model...")
model_manager = QuantizedModelManager()
model_manager.load_model()

# Apply pruning
print("Applying 30% pruning...")
pruner = ModelPruner(pruning_ratio=0.3, method="magnitude")
model = model_manager.llm.model
pruned_model = pruner.prune_model(model)
model_manager.llm.model = pruned_model

# Initialize agents
router = RouterAgent()
solver = SolverAgent(model_manager)
judge = JudgeAgent()
verifier = VerificationAgent()

# Evaluate question
question = "What is 15 + 27?"
gold_answer = "42"

# Route
routing = router.route(question, dataset="gsm8k")
print(f"Routed to: {routing.task_type}")

# Solve
response = solver.solve(question, routing)
print(f"Answer: {response.answer}")

# Judge
judgment = judge.judge_single(response.answer, gold_answer, "math")
print(f"Correct: {judgment.correct}")

# Verify with custom agent
verification = verifier.verify(response.answer, gold_answer, "math")
print(f"Verification: {verification}")

# Cleanup
model_manager.cleanup()

print("\nSystem Summary:")
print(f"- Model: Qwen3-4B-Thinking-2507")
print(f"- Quantization: INT4 (~2 GB)")
print(f"- Pruning: 30% of weights")
print(f"- Estimated VRAM: ~7-8 GB (vs 13 GB baseline)")
print(f"- Speed: ~2-3x faster")
print(f"- Accuracy: ~95% of baseline")
```

---

## Summary of Extension Points

| Extension | Files to Modify | Complexity | Impact |
|-----------|----------------|------------|--------|
| **New Agent** | `src/agents/*.py`, `evaluator.py` | Medium | Add new capability |
| **New Router** | `src/agents/router.py` or new file | Low-Medium | Better routing |
| **Quantization** | `src/models/*.py`, `config.py` | Medium | Reduce VRAM 50-75% |
| **Pruning** | `src/models/pruning.py` | High | Reduce VRAM 20-30% |
| **Batch Processing** | `src/utils/*.py` | Low | Improve throughput |
| **Caching** | `src/utils/cache.py` | Low | Speed up repeats |
| **New Task Type** | `router.py`, `solver.py`, `judge.py` | Medium | Support new datasets |
| **New Metrics** | `src/utils/metrics.py`, `judge.py` | Low | Better evaluation |

---

**End of Extension Guide**

For more examples and advanced techniques, see the README.md and inline code comments.
