# Multi-agent support ticket triage

A LangGraph pipeline that takes a raw customer support ticket and routes it
through four specialized nodes, producing either an auto-drafted reply grounded
in a real knowledge base, or a clean escalation with a stated reason.

Claude via the Anthropic API for the reasoning nodes, a persistent local
Chroma vector store, and local ONNX embeddings (so retrieval costs nothing and
runs offline).

```
ticket -> Classifier -> [abusive/out-of-scope?] ------------> ESCALATE
                            |
                            v
                        Retriever -> Responder -> Judge -> [SEND?] -> AUTO_REPLY
                                         ^          |
                                         +-- RETRY --+
                                                     |
                                                     +---------> ESCALATE
```

Two conditional edges make this a graph rather than a chain: the classifier
short-circuit (never spend retrieval and drafting on an abusive ticket) and the
judge's retry-or-escalate branch, with retries capped at one.

## Setup

```bash
brew install uv
uv venv --python 3.12
uv pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m triage.index        # build the Chroma collection
```

## Use

```bash
.venv/bin/python cli.py "Where do I download invoice PDFs?"
.venv/bin/python -m pytest tests/ -q    # unit tests, stubbed model, ~2s
.venv/bin/python -m pytest evals/ -v    # eval fixtures, real model
```

## Layout

| Path | What it is |
|---|---|
| `triage/state.py` | `TicketState`, the contract every node reads and writes |
| `triage/llm.py` | One call signature, per-role models, `anthropic` / `stub` |
| `triage/prompts.py` | Per-node system prompts and output schemas, in one place |
| `triage/nodes/` | The four nodes |
| `triage/graph.py` | Wiring and conditional edges |
| `triage/index.py` | Builds the Chroma collection from `triage/kb/` |
| `evals/fixtures.yaml` | Six tickets with declarative behavioral assertions |
| `evals/assertions.py` | The assertion engine, unit-tested separately |
| `docs/eval-log.md` | What the evals caught and what changed in response |

## Models, per role

Every model call goes through `triage/llm.py` and names the role making it.
Each role is bound to its own model:

| Role | Model | Why |
|---|---|---|
| Classifier | `claude-haiku-4-5` | Picks one of four fixed labels. No reasoning needed. |
| Responder | `claude-sonnet-5` | Rewrites retrieved text; thinking disabled. |
| Judge | `claude-sonnet-5` | Weighs four competing conditions; adaptive thinking on. |

Routing the mechanical call to the cheap model and reserving the expensive one
for the node whose mistakes are costly is a deliberate choice, and it makes the
per-role model a variable the eval suite can sweep:

```bash
# benchmark the judge on a stronger model, same fixtures
TRIAGE_MODEL_JUDGE=claude-opus-5 .venv/bin/python -m pytest evals/ -v

# no model at all, deterministic, free
TRIAGE_LLM=stub .venv/bin/python -m pytest tests/ -q
```

This is what makes "does this fixture fail because my prompt is wrong, or
because the model is too weak" an answerable question rather than a guess.

## Notes on the design

**Retrieval returns scores, not just text.** The retriever records
`max_similarity` so that low confidence is inspectable state rather than
something the model has to introspect about and report honestly.

**But a similarity threshold is a floor, not the answer.** Measured separation
between answerable and unanswerable fixtures is only 0.070, because cosine
similarity captures topical relatedness rather than whether a passage contains
the answer. Those come apart precisely when a question is on-topic but
unanswered — the most common hallucination trigger. See `docs/eval-log.md`
round 0. The threshold catches "nothing is even on-subject"; an explicit
grounding check handles the rest.

**Unit tests never touch a model.** Routing, chunking, and the assertion engine
run against a stub in about two seconds, so the control flow is CI-able and
model failures can't masquerade as logic bugs.
