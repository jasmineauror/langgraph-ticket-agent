#!/usr/bin/env python
"""Run one ticket through the triage graph and print the full trace.

    python cli.py "I was charged twice this month"
    python cli.py --file ticket.txt
    echo "..." | python cli.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from triage import llm
from triage.graph import run_ticket

RULE = "=" * 72


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(line, width=70, initial_indent=indent, subsequent_indent=indent)
        or indent
        for line in text.splitlines()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket", nargs="?", help="ticket text")
    parser.add_argument("--file", help="read ticket text from a file")
    args = parser.parse_args()

    if args.file:
        ticket_text = open(args.file).read()
    elif args.ticket:
        ticket_text = args.ticket
    elif not sys.stdin.isatty():
        ticket_text = sys.stdin.read()
    else:
        parser.error("provide ticket text, --file, or pipe it on stdin")

    ticket_text = ticket_text.strip()

    print(RULE)
    print(f"TICKET   (model: {llm.model_name()} via {llm.backend_name()})")
    print(RULE)
    print(_wrap(ticket_text))
    print()

    state = run_ticket(ticket_text)

    print(RULE)
    print("TRACE")
    print(RULE)
    for line in state.get("trace", []):
        print(_wrap(line, indent="  "))
    print()

    verdict = state.get("verdict")
    print(RULE)
    if verdict == "SEND":
        print("OUTCOME: AUTO_REPLY")
        print(RULE)
        print(_wrap(state.get("draft_reply", "")))
        sources = state.get("cited_sources") or []
        print()
        print(f"  cited sources: {', '.join(sources) if sources else '(none)'}")
    else:
        print("OUTCOME: ESCALATE")
        print(RULE)
        print(f"  category: {state.get('category')}")
        print(f"  reason:   {state.get('escalation_reason')}")
        draft = state.get("draft_reply")
        if draft:
            print()
            print("  rejected draft, for the human's context:")
            print(_wrap(draft, indent="    "))
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
