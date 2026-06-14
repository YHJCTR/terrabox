# Artifact Transition Evo Design

## Purpose

Build an experimental self-evolution method for Terrabox that stores and retrieves experience at the level of artifact state transitions, not complete trajectories or tool-name skills.

The first implementation is an offline MVP for the current OpenEarthAgent + EarthBench SFT and merged trajectory data. It must build a usable experience store, retrieve prompt context, and update `Q / N / Risk` statistics. It must leave clean interfaces for later online rollout updates and sub-agent verification.

## Problem

The current datasets have three properties that make full-trajectory memory weak:

1. Tool calls are long-tailed. Head tools such as calculator, solver, and perception tools dominate the corpus, while many EarthBench domain tools have few examples.
2. Full trajectory memories repeat common OpenEarth skeletons and can drown out low-frequency tool experience.
3. Tool execution success does not guarantee semantic correctness. A bbox, raster, or scalar can be syntactically valid but semantically wrong or incorrectly consumed downstream.

The method therefore treats the learnable unit as:

```text
current artifact state -> next artifact state
```

and records both signature-level and tool-level experience for that transition.

## Scope

### In Scope For MVP

- Create a new package: `src/terrabox/evolution/artifact_transition_evo/`.
- Build an offline experience store from `data/fixdata/sft_train_strict.jsonl` through the existing `FullSFTSample` loader.
- Provide an adapter boundary for shared `Trajectory` records so real rollout data can be added without changing the store design.
- Infer artifact signatures from tool slugs, argument names, and known Terrabox geospatial tool patterns.
- Store signature-level records keyed by:

```text
task_type + input_signature + output_signature
```

- Store tool-level records under each signature-level record keyed by:

```text
task_type + input_signature + output_signature + tool_slug
```

- Maintain `Q`, `N`, and `Risk` at both levels.
- Merge repeated head-tool experiences by updating statistics instead of appending duplicate experience text.
- Compute usage ranking:

```text
Q_use = lambda * Q_tool + (1 - lambda) * Q_sig
lambda = N_tool / (N_tool + k)
```

- Retrieve experience using a deterministic hybrid of BM25-lite lexical scoring, artifact signature matching, task type matching, `Q`, and `Risk`.
- Format prompt context that can be injected through `get_prompt_augmenter("artifact_transition_evo")`.
- Provide deterministic update APIs for feedback:

```text
Q <- Q + alpha * (r - Q)
Risk <- Risk + alpha_risk * (risk_observed - Risk)
N <- N + 1
alpha = alpha0 / sqrt(N + 1)
```

- Add tests for construction, deduplication, shrinkage, risk update, retrieval, prompt injection, and method registration.

### Out Of Scope For MVP

- No agent loop integration.
- No real LLM sub-agent call for silent-failure verification.
- No Milvus/vector dependency. The retriever exposes a boundary where dense retrieval can be added later.
- No changes to `src/terrabox/core/registry.py`, `src/terrabox/extensions.py`, or stable tool schemas.
- No new script under `scripts/`; the experimental CLI lives inside the new evolution package as `runner.py`.

## Architecture

The new package is organized by responsibility:

```text
artifact_transition_evo/
  __init__.py
  types.py
  signature.py
  builder.py
  store.py
  retriever.py
  feedback.py
  prompt_injector.py
  runner.py
  README.md
```

### `types.py`

Defines the stable data model:

- `ArtifactSignature`: normalized artifact kind plus optional role terms.
- `TransitionKey`: task type, input signature, output signature.
- `ToolKey`: transition key plus tool slug.
- `ToolExperience`: tool-specific text, `Q_tool`, `N_tool`, `Risk_tool`.
- `ExperienceRecord`: signature-level text, `Q_sig`, `N_sig`, `Risk_sig`, and a mapping of tool experiences.
- `FeedbackSignal`: reward, observed risk, optional reason, and optional score components.

### `signature.py`

Turns tool calls into coarse transition signatures.

The first version uses explicit rules for important Terrabox families:

- `geo_perception.instructsam`, `geo_perception.remotesam`, `geo_perception.strip_rcnn_detect`: image to bbox or detection set.
- `geo_perception.bbox_to_centroid`: bbox set to point set.
- `geo_perception.draw_bboxes`, `geo_perception.add_text`: image plus annotations to annotated image.
- `geo_raster.calculate_index`: raster bands to index raster.
- `geo_raster.compute_tvdi`: NDVI + LST raster to TVDI raster.
- `earth_sci.calculate_split_window`: band31 + band32 + emissivity rasters to LST raster or PWV raster.
- `earth_sci.calculate_lst_*`: thermal bands to LST raster.
- `geo_statistics.threshold_ratio`: raster to ratio scalar.
- `geo_statistics.batch_raster_stats`, `mean_of_means`, `max_value_and_index`, `min_value_and_index`: raster or numeric list to scalar or table.
- `compute.calculator`, `compute.solver`: numeric/text evidence to scalar or derived value.

Unknown tools fall back to argument-name heuristics:

- keys containing `image` -> `image`
- keys containing `bbox` -> `bbox_set`
- keys containing `band`, `raster`, `path`, `input_path`, `input_paths` -> `raster`
- keys containing `values`, `expression`, `threshold`, `ratio` -> `scalar`
- no evidence -> `unknown_artifact`

This is intentionally conservative and testable. It does not hard-code dataset-level gold trajectories.

### `builder.py`

Builds stores from existing data.

Primary path:

```python
build_store_from_sft(path: str | Path, limit: int | None = None) -> ArtifactTransitionStore
```

It uses `terrabox.evolution.full_shared.sft_schema.load_sft_samples`, reads `gold_tool_calls`, infers a transition for each tool call, and updates the store.

Secondary path:

```python
build_store_from_trajectories(trajectories: Iterable[Trajectory]) -> ArtifactTransitionStore
```

This path initially consumes `trajectory.tools_called`, `trajectory.turns`, `trajectory.artifacts`, and `trajectory.metadata["tool_trace"]` when available. It is the future online/rollout boundary.

### `store.py`

Owns persistence and statistics.

Responsibilities:

- Add or merge a transition.
- Add or merge a tool experience under a transition.
- Load/save store JSON.
- Return candidate records for retrieval.
- Compute `Q_use` for a tool under a transition.
- Keep duplicate head experiences compact by updating `N` and rolling `Q/R` instead of appending equivalent text.

The persisted file is a single JSON document for easy inspection:

```text
experience_store.json
```

The schema includes a version string so later migrations can be explicit.

### `retriever.py`

Implements deterministic hybrid retrieval:

```python
retrieve(query, task_type=None, current_signature=None, target_signature=None, top_k=5)
```

Scoring combines:

- lexical overlap between query and experience text
- task type match
- input/output signature match
- positive `Q_sig`
- negative `Risk_sig`

Tool ranking for a selected transition uses `Q_use`, then lower risk, then larger evidence count.

Dense vector retrieval is represented by a small protocol/interface but not implemented in the MVP.

### `feedback.py`

Implements update math and verification gating.

The MVP supports:

- update signature-level statistics
- update tool-level statistics
- decay step size by `N`
- decide whether online verification should be requested later:

```python
needs_verification(q_use, risk, n, *, q_threshold, risk_threshold, min_evidence)
```

This function is deterministic. Later online code can call an LLM sub-agent when it returns true.

### `prompt_injector.py`

Implements `PromptAugmenter`.

The prompt context should be compact and operational:

- retrieved artifact transition goals
- input/output constraints
- common downstream consumption constraints
- ranked tools under the selected transitions
- warnings for high-risk or low-evidence transitions

It should not include full trajectories by default.

### `runner.py`

Provides a package-local experiment CLI:

```bash
python -m terrabox.evolution.artifact_transition_evo.runner build-store \
  --sft data/fixdata/sft_train_strict.jsonl \
  --store-dir evolution_store/artifact_transition_evo \
  --limit 1000

python -m terrabox.evolution.artifact_transition_evo.runner retrieve \
  --store-dir evolution_store/artifact_transition_evo \
  --query "calculate high temperature area ratio from MODIS band31 band32 emissivity"
```

This avoids adding a new top-level `scripts/` file.

## Data Flow

### Offline Build

```text
SFT JSONL
  -> FullSFTSample
  -> gold_tool_calls
  -> transition signature inference
  -> ExperienceRecord / ToolExperience merge
  -> experience_store.json
```

Initial SFT records are treated as successful but not as perfect semantic proof. Default initial values:

```text
Q = 1.0
N = 1
Risk = 1.0 if argument_status is adapted_lossy or tool call is non-executable, else 0.0
```

If a sample has `all_gold_calls_executable = false`, the builder may still record a candidate with lower `Q` and higher `Risk`, but tests should focus on executable samples first.

### Retrieval

```text
task query + optional current/target signature
  -> retrieve signature-level candidates
  -> rank by lexical/signature/task/Q/R
  -> rank tools inside each transition by Q_use
  -> prompt context
```

### Future Online Update

The online rollout integration will later call:

```text
tool call + observation + FeedbackSignal
  -> infer actual transition
  -> update signature-level Q/N/R
  -> update selected tool-level Q/N/R
  -> if needs_verification then run sub-agent judge
```

The MVP only provides the APIs and deterministic gate.

## Error Handling

- Missing or malformed SFT rows are skipped with counts in the build manifest.
- Tool calls without a slug are skipped.
- Unknown signatures are retained as `unknown_artifact` instead of crashing.
- Store load validates the version field and raises a clear `ValueError` for incompatible files.
- Retrieval against an empty or missing store returns base prompt context instead of crashing.

## Testing

Tests live in:

```text
tests/test_artifact_transition_evo.py
```

Required test behaviors:

1. A minimal SFT row with `geo_perception.instructsam -> compute.solver -> compute.calculator` builds both signature-level and tool-level experiences.
2. Two identical tool calls merge into one record with `N > 1` instead of duplicated records.
3. `Q_use` shrinks toward `Q_sig` when `N_tool` is low and toward `Q_tool` when `N_tool` is high.
4. `Risk` updates with decayed step size when `risk_observed` is 1.
5. Retrieval prefers matching task type and artifact signature over unrelated text.
6. Prompt injector includes transition goals, tool ranking, and risk/evidence warnings.
7. `get_prompt_augmenter("artifact_transition_evo")` returns the new prompt injector.
8. Runner parser exposes `build-store` and `retrieve`.

Tests must be written before production code and must run without GPU, Docker, network, or LLM.

## Open Decisions Resolved

- First slice is offline MVP, not online agent integration.
- Store persistence is JSON for inspectability.
- Retriever is deterministic BM25-lite plus structured scoring; dense vectors are a future extension.
- Sub-agent verification is represented by interface and gating only in MVP.
- No new top-level scripts are added.

## Success Criteria

The MVP is complete when:

- The new package can build an experience store from a small SFT fixture and from a limited real SFT file.
- Unit tests pass locally.
- `get_prompt_augmenter("artifact_transition_evo", store_dir=...)` can retrieve and inject context.
- The implementation does not modify stable registry, extension, or toolkit interfaces.
- The design leaves a clear API for future online rollout feedback and sub-agent verification.
