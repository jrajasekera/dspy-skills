# DSPy Framework Documentation

Comprehensive reference for the DSPy framework - a declarative system for programming and optimizing LLM applications.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Signatures](#signatures)
- [Modules](#modules)
- [Adapters](#adapters)
- [Optimizers](#optimizers)
- [Retrieval & RAG](#retrieval--rag)
- [Evaluation](#evaluation)

---

## Core Concepts

DSPy abstracts LLM interaction into three layers:

1. **Signatures** - Declarative I/O specifications
2. **Modules** - Composable program building blocks
3. **Adapters** - Bridge between DSPy and LLM APIs

```python
import dspy

# Configure the LM globally
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

---

## Signatures

A **Signature** defines *what* a module does, not *how*. DSPy compiles it into an effective prompt.

### Inline Signatures

```python
# Simple
"question -> answer"

# With types
"sentence -> sentiment: bool"

# Multiple fields
"context: list[str], question: str -> answer: str"
```

### Class-based Signatures

```python
from typing import Literal
import dspy

class Emotion(dspy.Signature):
    """Classify the emotion of the sentence."""
    sentence: str = dspy.InputField(desc="The text to classify.")
    sentiment: Literal['sadness','joy','love','anger','fear','surprise'] = dspy.OutputField()
```

### Type Hints

Supported types:
- Basic: `str`, `int`, `float`, `bool`
- Collections: `list[str]`, `dict[str, int]`
- Optional: `Optional[str]`
- Literals: `Literal['a', 'b', 'c']`
- Pydantic models for complex structures
- `dspy.Reasoning` — a string-like type (DSPy 3.2.0) signaling a field carries step-by-step thinking; adapters can route it through native reasoning channels on reasoning-capable LMs

### Input Type Validation (DSPy 3.2.0)

DSPy 3.2.0 uses `typeguard` to check that call-time arguments match the declared signature. Mismatches emit a **warning** (opt-in via `dspy.settings.warn_on_type_mismatch=True`), not a hard error, so this is non-breaking. Separately, DSPy 3.2.0 raises `ValueError` at signature-construction time if input/output field names collide.

### Deprecated Field kwargs (DSPy 3.2.0)

`prefix`, `format`, and `parser` kwargs on `InputField` / `OutputField` now emit `DeprecationWarning`. They still work but will be removed in a future release. Adapter-era code should rely on field names + `desc` and let the adapter handle formatting.

---

## Modules

Modules are composable building blocks with learnable parameters.

### dspy.Predict

Basic prediction without reasoning:

```python
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence="It's a charming journey.")
print(response.sentiment)  # True
```

### dspy.ChainOfThought

Adds step-by-step reasoning:

```python
qa = dspy.ChainOfThought('question -> answer')
response = qa(question="What's special about ColBERT?")
print(response.reasoning)  # Step-by-step explanation
print(response.answer)
```

### dspy.ReAct

Agent with tool use:

```python
def search_wiki(query: str) -> str:
    """Search Wikipedia for information."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return results[0]['text']

def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return dspy.PythonInterpreter({}).execute(expression)

agent = dspy.ReAct("question -> answer: float", tools=[search_wiki, calculate])
result = agent(question="What is 100 divided by the year Columbus discovered America?")
```

---

## Adapters

Adapters translate between DSPy structures and LLM APIs.

### dspy.ChatAdapter (Default)

- Human-readable marker format: `[[ ## field_name ## ]]`
- Auto-fallback to JSONAdapter on failure
- Good for debugging

### dspy.JSONAdapter

- Native JSON output mode
- Lower latency, more reliable parsing
- Best for structured data

```python
# Explicitly set adapter
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter()
)
```

### dspy.XMLAdapter (DSPy 3.2.0)

- XML serialization/parsing for structured I/O
- Useful when a target model formats XML more reliably than JSON, or when you want compatibility across models with uneven JSON-mode support
- Same prompt-layer behavior as `JSONAdapter`; only format changes

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.XMLAdapter())
```

> DSPy 3.2.0 also adapts the adapter layer to depend on `BaseLM` rather than the concrete `LM` class. Custom `BaseLM` subclasses must now implement the model-capability checks (e.g. `supports_reasoning`) that used to be resolved via `litellm`.

---

## Optimizers

### Few-Shot Learning Optimizers

#### LabeledFewShot

Simplest optimizer - random selection from training set.

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(k=3)  # Use 3 examples
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `k` | Number of examples to include |

#### BootstrapFewShot

Generates demonstrations using a teacher model.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=dspy.evaluate.answer_exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    teacher_settings={'lm': dspy.LM("openai/gpt-4o")}
)
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Evaluation function | Required |
| `max_bootstrapped_demos` | Max generated demos | 4 |
| `max_labeled_demos` | Max labeled demos | 16 |
| `teacher_settings` | Teacher LM config | None |

#### BootstrapFewShotWithRandomSearch

Extends Bootstrap with random search over demo sets.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `num_candidate_programs` | Candidates to evaluate | 16 |
| `num_threads` | Parallel threads | 6 |

#### KNNFewShot

Dynamic example selection via k-NN.

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(k=3, trainset=trainset)
```

### Instruction Optimizers

#### COPRO

Iterative instruction refinement.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=metric,
    breadth=10,  # Initial candidates
    depth=3,     # Refinement iterations
    prompt_model=dspy.LM("openai/gpt-4o")
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `breadth` | Initial candidates | 10 |
| `depth` | Refinement iterations | 3 |
| `prompt_model` | LM for generation | Default LM |

#### MIPROv2

State-of-the-art: Bayesian optimization of instructions + demos.

> **Install note (DSPy 3.2.0+):** `optuna` moved to an optional extra. Install with `pip install "dspy[optuna]>=3.2.0"` or `MIPROv2` will raise an `ImportError`.

```python
import dspy

optimizer = dspy.MIPROv2(
    metric=dspy.evaluate.answer_exact_match,
    auto="medium",  # "light", "medium", "heavy"
    num_threads=24
)
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Evaluation metric | Required |
| `auto` | Preset config | None |
| `num_trials` | Optimization trials | Varies |
| `num_candidates` | Candidates per trial | 10 |
| `max_bootstrapped_demos` | Set 0 for instruction-only | 4 |
| `max_labeled_demos` | Labeled examples | 16 |
| `prompt_model` | LM for proposals | Default |
| `task_model` | LM for execution | Default |

**Auto presets:**
- `"light"`: ~10 trials, quick optimization
- `"medium"`: ~40 trials, balanced
- `"heavy"`: ~100+ trials, thorough

**Best for:** 200+ training examples, comprehensive tuning.

#### SIMBA

Identifies hard examples and generates improvement rules.

```python
from dspy.teleprompt import SIMBA

optimizer = SIMBA(metric=metric)
```

#### GEPA (Genetic-Pareto)

Newest optimizer: LLM reflection on full execution traces.

```python
import dspy

# GEPA requires a feedback metric with the full 5-arg signature
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    is_correct = gold.answer.lower() in pred.answer.lower()
    feedback = "Correct answer" if is_correct else f"Expected {gold.answer}, got {pred.answer}"
    return dspy.Prediction(score=float(is_correct), feedback=feedback)

optimizer = dspy.GEPA(
    metric=gepa_metric,
    reflection_lm=dspy.LM("openai/gpt-4o"),
    auto="medium"
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Signature `(gold, pred, trace, pred_name, pred_trace)`; returns `float` or `dspy.Prediction(score=..., feedback=...)` | Required |
| `reflection_lm` | Strong LM for reflection | Default LM |
| `auto` | "light", "medium", "heavy" | None |

> DSPy 3.2.0 removed GEPA's `enable_tool_optimization` (`ToolProposer`) — tool-description optimization is no longer supported.

**Best for:** Complex agentic systems, when you have rich textual feedback.

### Fine-tuning Optimizer

#### BootstrapFinetune

Distill a DSPy program into fine-tuned weights.

```python
import dspy

optimizer = dspy.BootstrapFinetune(
    metric=lambda gold, pred, trace=None: gold.label == pred.label
)
finetuned = optimizer.compile(
    program,
    trainset=trainset,
    train_kwargs={'learning_rate': 5e-5, 'num_train_epochs': 3}
)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Validation metric (optional) |
| `train_kwargs` | Training hyperparameters |

### Ensemble Optimizer

Combine multiple programs.

```python
from dspy.teleprompt import Ensemble

ensembled = Ensemble(reduce_fn=dspy.majority).compile([prog1, prog2, prog3])
```

### BetterTogether (DSPy 3.2.0 rewrite)

Composes arbitrary prompt- and weight-level optimizers via a strategy string:

```python
optimizer = dspy.BetterTogether(
    metric=metric,
    p=dspy.GEPA(metric=feedback_metric, reflection_lm=dspy.LM("openai/gpt-4o"), auto="light"),
    w=dspy.BootstrapFinetune(metric=metric),
)
compiled = optimizer.compile(program, trainset=trainset, strategy="p -> w -> p")
```

Defaults are `BootstrapFewShotWithRandomSearch` + `BootstrapFinetune` if `p`/`w` are omitted. Validation selection picks the best candidate across stages.

---

## Retrieval & RAG

### ColBERTv2 Setup

```python
import dspy

colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), rm=colbert)
```

### Embeddings retrievers (DSPy 3.2.0)

`dspy.EmbeddingsWithScores` (new) extends `Embeddings` to return similarity scores alongside passages:

```python
from dspy.retrievers import EmbeddingsWithScores

retriever = EmbeddingsWithScores(corpus=corpus, embedder=embedder, k=5)
result = retriever("What is the capital of France?")
# result.passages, result.indices, result.scores
```

Use this when you want to filter low-confidence context before generation or when your downstream module needs the similarity distribution to calibrate answers.

### RAG Module Pattern

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

---

## Evaluation

### Evaluate Class

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=8,
    display_progress=True
)
score = evaluator(program)
```

### Built-in Metrics

```python
# Exact match (normalized, case-insensitive)
dspy.evaluate.answer_exact_match

# F1 token overlap
dspy.evaluate.answer_passage_match
```

### SemanticF1

LLM-based semantic evaluation:

```python
from dspy.evaluate import SemanticF1

semantic_metric = SemanticF1()
result = semantic_metric(example, prediction)
score = result.score  # DSPy 3.2.0: returns dspy.Prediction, not a bare float
```

> **DSPy 3.2.0 behavioral change:** `SemanticF1` and `CompleteAndGrounded` return `dspy.Prediction(score=...)` rather than `float`. `dspy.Evaluate` handles this transparently via `.score`; only direct callers must be updated.

### Custom Metrics

```python
def my_metric(example, pred, trace=None):
    """Returns bool, int, or float."""
    return example.answer.lower() == pred.answer.lower()
```

---

## Errors & Caching (DSPy 3.2.0)

### `dspy.ContextWindowExceededError`

DSPy-owned exception raised by `BaseLM` subclasses when a prompt exceeds the LM's context window. Provider-agnostic replacement for string-matching on `litellm` errors:

```python
try:
    pred = module(question=very_long_question)
except dspy.ContextWindowExceededError:
    with dspy.context(lm=dspy.LM("anthropic/claude-opus-4-7")):
        pred = module(question=very_long_question)
```

### Safe disk-cache deserialization

`dspy.configure_cache(..., restrict_pickle=True, safe_types=[...])` restricts which object types DSPy deserializes from the disk cache. Recommended for any deployment that reads a shared or untrusted cache directory:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    disk_cache_dir="/var/cache/dspy",
    restrict_pickle=True,
    safe_types=[dspy.Prediction, dict, list, str, int, float, bool, type(None)],
)
```

### Safe module state loading

`BaseModule.load_state` filters unsafe LM-state keys (`api_base`, `base_url`, `model_list`) by default so that a saved module can't redirect a loader's LM calls to an attacker-controlled endpoint. Opt out with `allow_unsafe_lm_state=True` when loading trusted state.

---

## Best Practices

1. **Start simple** - Use `dspy.Predict` before `ChainOfThought`
2. **Measure first** - Establish baseline metrics before optimizing
3. **Small data works** - BootstrapFewShot works with ~10 examples
4. **Match optimizer to data** - MIPROv2 for 200+, Bootstrap for fewer
5. **Use strong teachers** - GPT-4 as teacher, GPT-3.5 as student
6. **Save compiled programs** - `program.save('optimized.json')`
