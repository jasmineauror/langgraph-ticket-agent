# The graph

Rendered from the compiled `StateGraph` itself, so it cannot drift from the
code the way a hand-drawn diagram would. Regenerate with:

```bash
TRIAGE_LLM=stub .venv/bin/python -m triage.diagram
```

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	classify(classify)
	retrieve(retrieve)
	respond(respond)
	judge(judge)
	auto_reply(auto_reply)
	escalate(escalate)
	__end__([<p>__end__</p>]):::last
	__start__ --> classify;
	classify -.-> escalate;
	classify -.-> retrieve;
	judge -.-> auto_reply;
	judge -.-> escalate;
	judge -.-> respond;
	respond --> judge;
	retrieve --> respond;
	auto_reply --> __end__;
	escalate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

The two decision points are what make this a graph rather than a chain:

- **After `classify`** -- an abusive or action-requiring ticket goes straight
  to `escalate`, skipping retrieval and drafting entirely. Cheapest correct
  behaviour for the worst input.
- **After `judge`** -- `SEND` to `auto_reply`, `RETRY` back to `respond`
  (capped at one attempt), or `ESCALATE` to a human.

Dotted edges are conditional; solid edges are unconditional.
