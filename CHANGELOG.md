# Changelog

All notable changes to DSPy Skills will be documented in this file.

## [Unreleased]

## [1.1.0] - 2026-04-22

### Added
- **dspy-react-agent-builder** - Build production ReAct agents with tools
- **dspy-debugging-observability** - MLflow tracing and production monitoring
- **dspy-output-refinement-constraints** - Output refinement with dspy.Refine and dspy.BestOfN
- **dspy-simba-optimizer** - Mini-batch Bayesian optimizer alternative
- **dspy-advanced-module-composition** - Ensemble and multi-chain composition patterns
- **dspy-custom-module-design** - Production-quality custom module design
- **dspy-optimize-anything** - Universal text artifact optimizer (GEPA optimize_anything API)
- New README sections: Agent Development, Output Validation, Advanced Patterns, Debugging & Monitoring
- Version compatibility metadata to all skills
- Cross-references between related skills
- LICENSE file (MIT)
- CONTRIBUTING.md with skill guidelines
- DSPy 3.2.0 coverage:
  - `dspy.XMLAdapter` (alternative to JSONAdapter/ChatAdapter) — in dspy-signature-designer
  - `dspy.EmbeddingsWithScores` (similarity-score retriever) — in dspy-rag-pipeline
  - `dspy.ContextWindowExceededError` — in dspy-debugging-observability
  - `dspy.Reasoning` type for reasoning-model signatures — in dspy-signature-designer and dspy-advanced-module-composition
  - Rewritten `dspy.BetterTogether` (arbitrary optimizers + strategy strings) — in dspy-finetune-bootstrap
  - `typeguard`-based input type validation (warn via `settings.warn_on_type_mismatch`) — in dspy-signature-designer
  - Safe `load_state` (filters unsafe LM-state keys) — in dspy-custom-module-design
  - `dspy.configure_cache(restrict_pickle=..., safe_types=...)` — in dspy-custom-module-design
  - `dspy.inspect_history(file=...)` file-output parameter — in dspy-debugging-observability

### Changed
- **dspy-haystack-integration** now uses progressive disclosure (references/ and examples/ directories)
- Updated workflow diagram in README to include validation and debugging steps
- Improved all skill descriptions with user trigger phrases for better discoverability
- Updated marketplace.json descriptions to match skill files
- Expanded skill count from 8 to 15 skills
- Bumped `dspy-compatibility` frontmatter on all 15 skills from `3.1.2` to `3.2.0`
- Installation instructions: `pip install dspy-ai>=3.1.2` → `pip install "dspy>=3.2.0"` (package name was always `dspy`)
- MIPROv2 skill now instructs `pip install "dspy[optuna]>=3.2.0"` (optuna moved to an optional extra in DSPy 3.2.0)
- `gepa[dspy]` recommendation bumped to `>=0.0.27` (eval-caching reduces metric call count)

### Fixed
- Skill descriptions now use trigger-based format instead of technical summaries
- GEPA metric examples across `dspy-gepa-reflective`, `dspy-evaluation-suite`, `docs/dspy-framework.md`, and `examples/code-snippets.py`: corrected long-standing bugs — signature is `(gold, pred, trace, pred_name, pred_trace)` (5 args, not 3) and return type must be `float` or `dspy.Prediction(score=..., feedback=...)` (not a `(score, feedback)` tuple)
- SIMBA metric contract documentation: removed incorrect claim that tuples are accepted
- Removed `enable_tool_optimization=True` from GEPA example in `dspy-react-agent-builder` (kwarg removed in DSPy 3.2.0)
- Removed `dspy.pretty_print_history` references (not a top-level export) and stale `dspy>=2.6.0` comment
- Fixed `SemanticF1` / `CompleteAndGrounded` usage: 3.2.0 returns `dspy.Prediction(score=...)`; direct-call code must read `.score`

## [1.0.0] - 2025-01-20

### Added
- Initial release with 8 DSPy skills
- 4 Optimizer skills (BootstrapFewShot, MIPROv2, GEPA, BootstrapFinetune)
- 4 Pipeline/Component skills (RAG, Signatures, Evaluation, Haystack Integration)
- Comprehensive DSPy framework documentation
- Production-ready code examples

[Unreleased]: https://github.com/OmidZamani/dspy-skills/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/OmidZamani/dspy-skills/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/OmidZamani/dspy-skills/releases/tag/v1.0.0
