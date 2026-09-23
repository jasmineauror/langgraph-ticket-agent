"""Render the graph to docs/graph.md as Mermaid.

    TRIAGE_LLM=stub python -m triage.diagram

Generated from the compiled StateGraph rather than drawn by hand, so the
picture cannot drift away from the wiring the way a hand-maintained diagram
does. (TRIAGE_LLM=stub only avoids constructing a live API client; no model is
called -- compiling the graph does not run it.)
"""

from __future__ import annotations

import pathlib

from .graph import build_graph

OUT = pathlib.Path(__file__).parent.parent / "docs" / "graph.md"

PREAMBLE = """# The graph

Rendered from the compiled `StateGraph` itself, so it cannot drift from the
code the way a hand-drawn diagram would. Regenerate with:

```bash
TRIAGE_LLM=stub .venv/bin/python -m triage.diagram
```

"""

NOTES = """
The two decision points are what make this a graph rather than a chain:

- **After `classify`** -- an abusive or action-requiring ticket goes straight
  to `escalate`, skipping retrieval and drafting entirely. Cheapest correct
  behaviour for the worst input.
- **After `judge`** -- `SEND` to `auto_reply`, `RETRY` back to `respond`
  (capped at one attempt), or `ESCALATE` to a human.

Dotted edges are conditional; solid edges are unconditional.
"""


def main() -> None:
    mermaid = build_graph().get_graph().draw_mermaid()
    OUT.write_text(f"{PREAMBLE}```mermaid\n{mermaid}```\n{NOTES}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
