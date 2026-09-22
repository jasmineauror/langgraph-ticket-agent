# Eval log

A record of what the eval suite caught and what changed in response. Each round
is one hypothesis, one change, one rerun.

---

## Round 0 — Retrieval calibration (before any model call)

Before choosing a grounding threshold, I measured what the retriever actually
returns for each fixture. No LLM involved; this is pure vector search over the
24 indexed chunks.

| Fixture | max_similarity | Top source |
|---|---|---|
| normal-billing | 0.621 | billing-invoices.md |
| technical-in-kb | 0.587 | tech-api-rate-limits.md |
| abusive | 0.253 | tech-webhooks.md |
| refund-request | 0.611 | billing-payment-methods.md |
| **no-kb-answer** | **0.517** | tech-integrations.md |
| ambiguous-two-issues | 0.629 | tech-webhooks.md |

**Finding: the separation is 0.070.** The worst answerable fixture scores 0.587;
the fixture the knowledge base genuinely cannot answer scores 0.517.

This is a problem, and it invalidates part of my own design note. The spec said
"not-knowing lives in code" — that a retrieval distance threshold makes
not-knowing a measurable property rather than something the model has to
introspect about. The measurement says that only half works.

**Why the gap is so small.** The `no-kb-answer` ticket asks about self-hosted /
on-premise deployment. The nearest chunk is `tech-integrations.md`, which
discusses data sources, destinations, and deployment-adjacent vocabulary. The
embedding model correctly reports that these two texts are *about similar
things*. They are. The knowledge base still contains no answer.

Cosine similarity measures **topical relatedness, not answer presence.** Those
come apart exactly when a question is on-topic but unanswered — which is the
most common real-world hallucination trigger, not an edge case.

**What a threshold can and cannot do.** A threshold at 0.55 would pass all six
fixtures today. It would also be fitted to six hand-written examples with 0.070
of headroom, which is not a threshold, it is a coincidence with a number
attached. What the scores *do* support is a much lower floor: at 0.253, the
abusive ticket is genuinely unrelated to everything in the knowledge base, and
any cutoff in the 0.35-0.45 range separates "nothing remotely relevant" from
"something on-topic" robustly.

**Decision.** Two gates rather than one, each doing the job it is actually good at:

1. **Code gate, set low (0.40).** Catches "nothing in the knowledge base is even
   on the same subject." Robust, cheap, no model call, and not fitted to the
   fixtures.
2. **Explicit grounding check in the model's hands.** For the on-topic-but-
   unanswered case, ask directly whether the retrieved passages contain the
   specific fact requested. This is a question about text the model can see, not
   a question about its own knowledge — which is the difference between a check
   that works and one that doesn't.

The threshold stays in the design, demoted from primary mechanism to floor.

---
