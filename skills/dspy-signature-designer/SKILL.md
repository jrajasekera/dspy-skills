---
name: dspy-signature-designer
version: "1.0.0"
dspy-compatibility: "3.2.0"
description: This skill should be used when the user asks to "create a DSPy signature", "define inputs and outputs", "design a signature", "use InputField or OutputField", "add type hints to DSPy", mentions "signature class", "type-safe DSPy", "Pydantic models in DSPy", or needs to define what a DSPy module should do with structured inputs and outputs.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Signature Designer

## Goal

Design clear, type-safe signatures that define what your DSPy modules should do.

## When to Use

- Defining new DSPy modules
- Need structured/validated outputs
- Complex input/output relationships
- Multi-field responses

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `task_description` | `str` | What the module should do |
| `input_fields` | `list` | Required inputs |
| `output_fields` | `list` | Expected outputs |
| `type_constraints` | `dict` | Type hints for fields |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `signature` | `dspy.Signature` | Type-safe signature class |

## Workflow

### Inline Signatures (Simple)

```python
import dspy

# Basic
qa = dspy.Predict("question -> answer")

# With types
classify = dspy.Predict("sentence -> sentiment: bool")

# Multiple fields
rag = dspy.ChainOfThought("context: list[str], question: str -> answer: str")
```

### Class-based Signatures (Complex)

```python
from typing import Literal, Optional
import dspy

class EmotionClassifier(dspy.Signature):
    """Classify the emotion expressed in the text."""
    
    text: str = dspy.InputField(desc="The text to analyze")
    emotion: Literal['joy', 'sadness', 'anger', 'fear', 'surprise'] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
```

## Type Hints Reference

```python
from typing import Literal, Optional, List
from pydantic import BaseModel

# Basic types
field: str = dspy.InputField()
field: int = dspy.OutputField()
field: float = dspy.OutputField()
field: bool = dspy.OutputField()

# Collections
field: list[str] = dspy.InputField()
field: List[int] = dspy.OutputField()

# Optional
field: Optional[str] = dspy.OutputField()

# Constrained
field: Literal['a', 'b', 'c'] = dspy.OutputField()

# Pydantic models
class Person(BaseModel):
    name: str
    age: int

field: Person = dspy.OutputField()
```

## Production Examples

### Summarization

```python
class Summarize(dspy.Signature):
    """Summarize the document into key points."""
    
    document: str = dspy.InputField(desc="Full document text")
    max_points: int = dspy.InputField(desc="Maximum bullet points", default=5)
    
    summary: list[str] = dspy.OutputField(desc="Key points as bullet list")
    word_count: int = dspy.OutputField(desc="Total words in summary")
```

### Entity Extraction

```python
from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int

class ExtractEntities(dspy.Signature):
    """Extract named entities from text."""
    
    text: str = dspy.InputField()
    entity_types: list[str] = dspy.InputField(
        desc="Types to extract: PERSON, ORG, LOC, DATE",
        default=["PERSON", "ORG", "LOC"]
    )
    
    entities: List[Entity] = dspy.OutputField()
```

### Multi-Label Classification

```python
class MultiLabelClassify(dspy.Signature):
    """Classify text into multiple categories."""
    
    text: str = dspy.InputField()
    
    categories: list[str] = dspy.OutputField(
        desc="Applicable categories from: tech, business, sports, entertainment"
    )
    primary_category: str = dspy.OutputField(desc="Most relevant category")
    reasoning: str = dspy.OutputField(desc="Explanation for classification")
```

### RAG with Confidence

```python
class GroundedAnswer(dspy.Signature):
    """Answer questions using retrieved context with confidence."""
    
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    question: str = dspy.InputField()
    
    answer: str = dspy.OutputField(desc="Factual answer from context")
    confidence: Literal['high', 'medium', 'low'] = dspy.OutputField(
        desc="Confidence based on context support"
    )
    source_passage: int = dspy.OutputField(
        desc="Index of most relevant passage (0-based)"
    )
```

### Complete Module with Signature

```python
import dspy
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)

class AnalyzeSentiment(dspy.Signature):
    """Analyze sentiment with detailed breakdown."""
    
    text: str = dspy.InputField(desc="Text to analyze")
    
    sentiment: Literal['positive', 'negative', 'neutral', 'mixed'] = dspy.OutputField()
    score: float = dspy.OutputField(desc="Sentiment score from -1 to 1")
    aspects: list[str] = dspy.OutputField(desc="Key aspects mentioned")
    reasoning: str = dspy.OutputField(desc="Explanation of sentiment")

class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeSentiment)
    
    def forward(self, text: str):
        try:
            result = self.analyze(text=text)
            
            # Validate score range
            if hasattr(result, 'score'):
                result.score = max(-1, min(1, float(result.score)))
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return dspy.Prediction(
                sentiment='neutral',
                score=0.0,
                aspects=[],
                reasoning="Analysis failed"
            )

# Usage
analyzer = SentimentAnalyzer()
result = analyzer(text="The product quality is great but shipping was slow.")
print(f"Sentiment: {result.sentiment} ({result.score})")
print(f"Aspects: {result.aspects}")
```

## Best Practices

1. **Descriptive docstrings** - The class docstring becomes the task instruction
2. **Field descriptions** - Guide the model with `desc` parameter
3. **Constrain outputs** - Use `Literal` for categorical outputs
4. **Default values** - Provide sensible defaults for optional inputs
5. **Validate types** - Pydantic models ensure structured output

## Advanced Field Options

```python
# Pydantic-style validation constraints
class ConstrainedSignature(dspy.Signature):
    """Example with validation constraints."""

    text: str = dspy.InputField(
        min_length=5,
        max_length=100,
        desc="Input text between 5-100 chars"
    )
    number: int = dspy.InputField(
        gt=0,
        lt=10,
        desc="Number between 0 and 10"
    )
    score: float = dspy.OutputField(
        ge=0.0,
        le=1.0,
        desc="Score between 0 and 1"
    )
    count: int = dspy.OutputField(
        multiple_of=2,
        desc="Even number count"
    )
```

> **Deprecated in DSPy 3.2.0:** `prefix`, `format`, and `parser` kwargs on `InputField` / `OutputField` now emit `DeprecationWarning`. They still work but will be removed in a later release. Prefer descriptive field names + `desc` over custom prefix/format; adapters handle the formatting.

## Input Type Validation (DSPy 3.2.0)

DSPy 3.2.0 uses `typeguard` to check that inputs passed to a module match the field annotations declared on its signature. Mismatches produce a **warning**, not a hard error — so this is non-breaking for existing code:

```python
import dspy

dspy.settings.configure(warn_on_type_mismatch=True)  # default is off; opt-in

class Classify(dspy.Signature):
    text: str = dspy.InputField()
    label: bool = dspy.OutputField()

classify = dspy.Predict(Classify)
classify(text=123)  # typeguard logs a warning: expected str, got int
```

> Separately, DSPy 3.2.0 **does** raise `ValueError` at signature construction time if you declare duplicate input/output field names (previously silently shadowed).

## XMLAdapter (DSPy 3.2.0)

If your target LM formats structured output more reliably as XML than JSON, swap `JSONAdapter` / `ChatAdapter` for `dspy.XMLAdapter`:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.XMLAdapter())
```

Use when JSON-mode models degrade on deeply nested outputs or when you want wider model compatibility. Prompt-layer structure is identical — only the serialization/parsing format changes.

## `dspy.Reasoning` type (DSPy 3.2.0)

A new `dspy.Reasoning` type is exported for signatures targeting reasoning models (o-series, etc.):

```python
class AnalyzedAnswer(dspy.Signature):
    """Answer a question with reasoning trace."""
    question: str = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField(desc="Step-by-step thinking")
    answer: str = dspy.OutputField()
```

`dspy.Reasoning` behaves like a string but carries semantics that let adapters route it through native-reasoning channels when the LM supports them. Note that in 3.2.0 `dspy.ChainOfThought` does **not** auto-switch to native reasoning — that behavior was added then reverted before release. Use `dspy.Reasoning` explicitly in a signature when you want the typed hook.

## Limitations

- Complex nested types require Pydantic models
- Some LLMs struggle with strict type constraints
- Field descriptions and constraints add to prompt length
- Default values only work for InputField, not OutputField

## Official Documentation

- **DSPy Documentation**: https://dspy.ai/
- **DSPy GitHub**: https://github.com/stanfordnlp/dspy
- **Signatures API**: https://dspy.ai/api/signatures/
- **Signatures Guide**: https://dspy.ai/learn/programming/signatures/
