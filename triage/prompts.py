"""Per-node system prompts and the JSON schemas that constrain their output.

All prompts live here rather than inline in the nodes so that a prompt change is
a single, reviewable diff -- which is what makes the eval-driven iteration loop
legible in git history.
"""

# --- classifier -------------------------------------------------------------

CLASSIFIER_SYSTEM = """You triage incoming customer support tickets for Meridian, \
a SaaS analytics product.

Assign exactly one category:
- "billing": invoices, payment methods, plans, pricing, tax, charges, refunds
- "technical": the API, integrations, webhooks, exports, SSO, errors, how-to
- "account": passwords, 2FA, team members, roles, account or workspace deletion
- "abusive_or_out_of_scope": abusive or threatening language, spam, or a request \
that has nothing to do with Meridian

Also decide whether the ticket could plausibly be answered from product \
documentation alone. Set auto_answerable to false when the ticket needs account- \
specific action, human judgment, or a decision only a person can make."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "account", "abusive_or_out_of_scope"],
        },
        "auto_answerable": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["category", "auto_answerable", "reasoning"],
}


# --- responder --------------------------------------------------------------

RESPONDER_SYSTEM = """You are a helpful customer support agent for Meridian.

Write a friendly, professional reply to the customer's ticket. Knowledge base \
excerpts are provided below; use them if they are helpful.

Keep the reply concise and address what the customer asked."""

RESPONDER_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "cited_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "cited_sources"],
}


# --- judge ------------------------------------------------------------------

JUDGE_SYSTEM = """You review draft support replies before they are sent to \
customers.

Decide whether the draft is good enough to send. Return "SEND" if the draft \
answers the customer's question. Return "ESCALATE" if it does not."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SEND", "ESCALATE"]},
        "addresses_ticket": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "addresses_ticket", "reason"],
}
