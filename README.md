# Multi-agent support ticket triage

A LangGraph pipeline that takes a raw customer support ticket and routes it
through four specialized nodes, producing either an auto-drafted reply grounded
in a real knowledge base, or a clean escalation with a stated reason.

Gemini for the reasoning nodes, a persistent local Chroma vector store, and
local ONNX embeddings. Runs on Gemini's free tier, so the whole project costs
nothing: retrieval is local, and the three model calls per ticket are free-tier
requests.

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

## Results

```
EVAL SCORE: 7/7 passed
  [PASS ] normal-billing          [PASS ] refund-request-action
  [PASS ] technical-in-kb         [PASS ] refund-policy-question
  [PASS ] abusive                 [PASS ] no-kb-answer
                                  [PASS ] ambiguous-two-issues
```

Plus 30 unit tests in 0.3s against a stubbed model.

That score took three rounds of prompt iteration and **five rounds where the
harness turned out to be measuring itself** — a DNS outage that looked like six
prompt failures, four fixtures passing without ever reaching the retriever, a
429 misread as a 503 so the suite disabled its own fallbacks, and a reporter of
mine that printed "3/3 passed" while four of seven fixtures never ran.

[docs/eval-log.md](docs/eval-log.md) records every round with the actual
hallucinated drafts, the hypothesis, the change, and the new score. It is the
most useful file in the repo.

## Setup

```bash
brew install uv
uv venv --python 3.12
uv pip install -r requirements.txt

export GEMINI_API_KEY=...        # free, no card: aistudio.google.com/apikey
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
| `triage/llm.py` | One call signature, per-role models, `gemini` / `stub` |
| `triage/prompts.py` | Per-node system prompts and output schemas, in one place |
| `triage/nodes/` | The four nodes |
| `triage/graph.py` | Wiring and conditional edges |
| `triage/index.py` | Builds the Chroma collection from `triage/kb/` |
| `evals/fixtures.yaml` | Seven tickets with declarative behavioral assertions |
| `evals/assertions.py` | The assertion engine, unit-tested separately |
| `docs/eval-log.md` | What the evals caught and what changed in response |

## Models, per role

Every model call goes through `triage/llm.py` and names the role making it.
Each role is bound to its own model:

| Role | Model (first choice) | Thinking | Why |
|---|---|---|---|
| Classifier | `gemini-3.5-flash-lite` | `MINIMAL` | Picks one of four fixed labels. |
| Responder | `gemini-3.5-flash` | `LOW` | Rewrites retrieved text. |
| Judge | `gemini-3-flash-preview` | `HIGH` | Weighs four competing conditions; its mistakes reach the customer. |

Each role walks an ordered **fallback chain**, and the three roles lead with
three *different* models on purpose: the free tier caps requests at **20 per day
per model**, so two roles sharing a first choice halves the number of eval runs
a day's quota supports. `triage/llm.py` holds the chains.

Routing the mechanical call to the cheap model and reserving the expensive one
for the node whose mistakes are costly is a deliberate choice, and it makes the
per-role model a variable the eval suite can sweep:

```bash
# benchmark the judge on a stronger model, same fixtures
TRIAGE_MODEL_JUDGE=gemini-2.5-pro .venv/bin/python -m pytest evals/ -v

# no model at all, deterministic, no network
TRIAGE_LLM=stub .venv/bin/python -m pytest tests/ -q
```

### Backends

| `TRIAGE_LLM` | Models | Free-tier budget |
|---|---|---|
| `gemini` (default) | `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3-flash-preview` | **20 requests/day/model** |
| `groq` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` | **1,000 requests/day/model** |
| `stub` | none | unlimited, no network |

A run costs ~15 requests, so Gemini's free tier supports roughly **4 runs a
day** and Groq's roughly **66**. Groq is the better choice for iterating; both
support *strict* schema-constrained decoding, so the shape guarantee the nodes
rely on is identical either way.

```bash
export GROQ_API_KEY=gsk_...    # free, no card: console.groq.com/keys
TRIAGE_LLM=groq .venv/bin/python -m pytest evals/ -v
```

Adding Groq touched only `triage/llm.py`. No node, no prompt, and no fixture
changed — which is the adapter earning its keep.

## The console

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit
.venv/bin/streamlit run streamlit_app.py
```

Three pages behind a passcode:

| Page | What it shows |
|---|---|
| **Triage Console** | A ticket streaming through the four stages, with retrieval scores against the grounding floor, the judge's rubric, and which model served each role |
| **Eval Suite** | The seven fixtures run live, with per-assertion results |
| **Knowledge Base** | All 15 documents and 24 chunks, plus a similarity probe you can point at any query |

The console is built to show the decisions, not hide them. A chat box would
conceal exactly the parts worth discussing: that `abusive` never reaches the
retriever, that `no-kb-answer` scores 0.517 — high enough to look relevant,
too low to answer — and that the responder declines rather than inventing.

### Demo mode

A sidebar toggle runs the whole pipeline with **no API calls and no
credentials**:

```bash
TRIAGE_LLM=demo pytest evals/ -q      # 7/7 in under a second, offline
```

Routing, vector search, both code gates and the judge rubric all execute for
real; only the model's wording is replayed, from responses recorded off a live
run by `python -m triage.record_demo`. Keys are a hash of the full prompt, so a
changed prompt or a rebuilt index *misses* and is labelled generic rather than
silently returning a reply recorded for different input. A ticket outside the
recorded set gets a conservative canned response and the UI says so.

It exists because a live demo that depends on a free-tier quota is a demo that
can fail in front of someone.

### The passcode is not authentication

It is a shared secret over HTTPS with no server-side rate limiting — enough to
stop a crawler or a forwarded link draining a daily model quota, and nothing
more. Don't put anything behind it you'd mind a determined person reading.

Structured output is enforced by the API via `response_schema`, which
constrains decoding. Malformed JSON is therefore not a failure mode the prompts
have to defend against.

One trap worth knowing: **thinking tokens draw from the same budget as the
visible answer.** A `max_output_tokens` sized only for the reply returns *empty
text* rather than an error when the model thinks. `MAX_OUTPUT_TOKENS` in
`triage/llm.py` is sized for thinking plus answer, and an empty response raises
with the finish reason attached.

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
