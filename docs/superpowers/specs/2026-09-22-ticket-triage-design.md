# Multi-Agent Support Ticket Triage — Design

**Date:** 2026-09-22
**Status:** Approved, in implementation
**Time budget:** ~3 hours

## Purpose

A LangGraph pipeline that takes a raw customer support ticket and routes it through
four specialized nodes, producing either an auto-drafted reply grounded in a real
knowledge base, or a clean escalation with a stated reason.

This is interview-prep for a role whose JD names LangGraph, vector DBs, prompt
engineering, and model evaluation. It is structurally the same shape as an
Extractor -> Matcher -> Reconciler pipeline built at Inovalon: specialized stages,
explicit state, conditional routing.

Two parts carry the weight and get real attention, not minimal effort:

1. A **real vector DB** (Chroma, persistent on disk) rather than an in-memory toy.
2. A **write -> eval -> fail -> iterate loop**, executed and recorded, not described.

Non-goals: no UI, no hosted infra, no scraped data, no deployment.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Runtime | Python 3.12 via `uv` | System Python is 3.9.6, too old for current LangGraph/Chroma |
| Graph | `langgraph`, `langchain-core` | The thing being practiced |
| Vector DB | `chromadb`, persistent at `./chroma_db` | Real collections, metadata filters, distance scores |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` | Offline, free, adequate for a ~15-doc KB |
| LLM | Claude API, per-role models | Haiku 4.5 classifies, Sonnet 5 responds and judges; reliable structured output and strong refusal behavior |
| Eval harness | `pytest` | Fixture-per-test, familiar output |

**Cost: well under $1 for the whole build** (~$0.05 per 6-fixture eval run).
Embeddings and the vector store are local and free; only the three reasoning
calls per ticket are billed.

**Revised 2026-09-22:** originally specified a local Ollama model for $0. Changed
to the API after the 9GB model pull proved to be the dominant cost in wall-clock
time rather than dollars, and because reliable refusal behavior is the single
property this project is built to demonstrate.

### LLM adapter

All model calls go through one `call(role, ...)` signature in `triage/llm.py`.
Each role is bound to its own model, overridable per role from the environment.
Rationale:

- If a fixture fails for reasons unrelated to prompt quality, the adapter isolates
  "my prompt is wrong" from "this model is too small" — otherwise the eval signal is
  ambiguous and the iteration loop teaches nothing.
- Running the same eval suite with a different model in one role is a real
  benchmark, and the cheapest way to tell a prompt problem from a capability
  problem.
- Binding the classifier to Haiku and the judge to Sonnet is itself a cost
  decision worth defending: the cheap model handles the four-label choice, the
  expensive one handles the call whose errors reach a customer.

Cost of the abstraction is ~30 lines.

## Architecture

```
ticket -> Classifier -> [abusive/out-of-scope?] -----------------> ESCALATE
                            |
                            v
                        Retriever -> Responder -> Judge -> [send?] -> AUTO_REPLY
                                         ^                   |
                                         +---- retry (max 1) -+
                                                             |
                                                             +---> ESCALATE
```

Two conditional edges make this a graph rather than a chain: the classifier
short-circuit and the judge's retry/escalate branch.

### State

`TicketState` (TypedDict) is the single contract between nodes. Each node reads
declared keys and writes declared keys; no node reaches into another's internals.

```
ticket_text        str           input
category           str           billing | technical | account | abusive_or_out_of_scope
auto_answerable    bool
retrieved          list[Chunk]   text + source + distance
max_similarity     float         drives the "KB has no answer" gate
draft_reply        str
cited_sources      list[str]
verdict            str           SEND | RETRY | ESCALATE
escalation_reason  str
retry_count        int           capped at 1
trace              list[str]     per-node log, printed by the CLI
```

### Nodes

**Classifier** (`nodes/classifier.py`) — tags the ticket and sets `auto_answerable`.
On `abusive_or_out_of_scope` the graph routes straight to escalation, skipping
retrieval and drafting entirely. Cheapest correct behavior for the worst input.

**Retriever** (`nodes/retriever.py`) — embeds the ticket, queries Chroma, returns
chunks **with distance scores**, and records `max_similarity`. Returning scores rather
than just text is what makes "the KB has no answer" a measurable property of
retrieval instead of something the model has to introspect about.

**Responder** (`nodes/responder.py`) — drafts a reply from retrieved context only.
Instructed to emit an explicit refusal marker when nothing retrieved is relevant,
rather than answering from general knowledge.

**Judge** (`nodes/judge.py`) — a four-part check:
1. Does the draft address the ticket?
2. Is `max_similarity` below the grounding threshold?
3. Does the draft cite a retrieved source?
4. Does it touch a sensitive area (refunds, legal, abuse)?

Routes `SEND`, `RETRY` (once), or `ESCALATE` with a reason.

### Design note: not-knowing lives in code

The refusal decision is deliberately split between a **code gate** (the retrieval
distance threshold) and **model judgment**, not left to the model alone. Small models
are trained toward helpfulness and will fill a knowledge gap when asked. Making low
retrieval confidence a hard, inspectable condition means a weaker model degrades
gracefully instead of confidently hallucinating.

### Terminal states

- `AUTO_REPLY` — draft plus cited sources.
- `ESCALATE` — reason, category, and the rejected draft, so the human has context.

## Knowledge base

~15 hand-written markdown docs under `triage/kb/`, covering billing, technical, and
account topics. Deliberately incomplete: at least one fixture asks something the KB
genuinely cannot answer, so the no-hallucination assertion tests real behavior rather
than a contrived gap.

`triage/index.py` builds/rebuilds the Chroma collection. Idempotent.

## Eval layer

`evals/fixtures.yaml` — six hand-written tickets with declarative assertions:

| Fixture | Assertions |
|---|---|
| Normal billing question | `must_auto_reply`, `must_cite_source` |
| Angry / abusive | `must_escalate`, reason `abusive` |
| Refund request | `must_escalate` (sensitive), `must_not_auto_reply` |
| Question with no KB answer | `must_escalate`, `must_not_contain` invented specifics |
| Ambiguous two-issue ticket | asserted explicitly, either escalate or address both |
| Technical question in KB | `must_auto_reply`, `must_cite_source` naming the doc |

`evals/test_evals.py` runs one pytest case per fixture. Assertion vocabulary:
`must_escalate`, `must_auto_reply`, `must_not_auto_reply`, `must_cite_source`,
`must_not_contain`, `expected_reason`.

## The deliberate-failure loop

The Responder and Judge prompts are written **badly on purpose first** and committed
that way:

- Responder without the grounding constraint — expected to invent a refund policy on
  the no-KB-answer fixture.
- Judge without the sensitive-topics check — expected to send the refund ticket
  instead of escalating.

Then: run evals, capture failures, fix the prompts, rerun. Output is a git history
showing the iteration plus `docs/eval-log.md` recording, per round: what failed, the
hypothesis, the prompt change, and the new score.

The log is the artifact, not a nicety. Walking an interviewer through a real
iteration is more convincing than presenting a finished thing that "worked first try."

## Testing

- **Unit** — node-level tests against a stubbed LLM. Fast, deterministic, no model
  required; covers graph routing and state transitions. Keeps routing logic CI-able.
- **Integration** — the eval fixtures, against the real model.

## Repo layout

```
triage/
  state.py          TicketState
  llm.py            adapter: OLLAMA | ANTHROPIC
  prompts.py        per-node system prompts, versioned in one place
  nodes/
    classifier.py  retriever.py  responder.py  judge.py
  graph.py          wiring + conditional edges
  index.py          build the Chroma collection
  kb/               ~15 markdown docs
cli.py              run one ticket, print the trace
evals/
  fixtures.yaml  test_evals.py
docs/
  eval-log.md       the iteration record
```

## Risks

| Risk | Mitigation |
|---|---|
| Model emits malformed JSON | Schema is injected into the system prompt; the adapter recovers JSON from prose and retries once |
| A fixture fails from model capability, not prompt quality | Raise that role's model via `TRIAGE_MODEL_<ROLE>`; the comparison is itself a deliverable |
| API spend runs away during iteration | Per-role `max_tokens` ceilings; thinking enabled only on the judge; ~$0.05 per full eval run |
| Judge retry loop spins | `retry_count` hard-capped at 1 |
