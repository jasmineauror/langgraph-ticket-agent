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

## Round 1 — Baseline: the naive prompts, measured

First clean run. All 7 fixtures executed, no infrastructure errors.

**Score: 5/7.** Both failures were real, and both were the designed ones.

### `no-kb-answer` — confident hallucination, zero citations

max_similarity 0.517, `cited_sources: []`, outcome AUTO_REPLY:

> "Currently, Meridian is offered exclusively as a cloud SaaS solution, and we
> do not provide a self-hosted or on-premise deployment option at this time."

Nothing in the knowledge base says this. The model asserted a fact about product
availability, cited nothing, and the judge sent it.

Worth noting how it evaded the content assertions: the forbidden-substring list
was written for *positive* hallucination (`$`, "yes, we offer"). It hallucinated
in the **negative** direction — "we do not provide" — and only `must_escalate`
caught it. A claim about what a product lacks is just as much a claim.

### `refund-policy-question` — right sentence, wrong question

> "any remaining prepaid period is handled on a case-by-case basis and is not
> automatically credited."

That sentence is genuinely in the knowledge base — in `account-deletion.md`,
about deleting a workspace. The ticket asked about cancelling an annual plan.
Retrieval surfaced a real passage and the responder applied it to a different
question. Technically sourced, contextually wrong, cited as nothing.

This failure mode is more dangerous than invention, because it survives a
"is this grounded in the sources?" check answered carelessly. The passage *is*
in the sources. It just does not answer what was asked.

**Hypotheses:**
1. The responder prompt says to use excerpts "if they are helpful" — permission
   to ignore them, with no refusal path and no citation requirement.
2. The judge checks only `addresses_ticket`. Both bad drafts *did* address the
   ticket. The check was satisfied by drafts that were wrong.
3. The grounding floor decided in round 0 was never implemented. Documented as
   settled, absent from the code.

---

## Round 2 — Fix the prompts and build the gate

**Changes:**

*Responder* — five ordered rules replacing "use them if they are helpful":
every claim traceable to an excerpt; **a negative claim needs evidence exactly
as much as a positive one**; check the excerpt answers *the question asked*, not
a related one; emit `INSUFFICIENT_CONTEXT` when the excerpts fall short, framed
as a correct outcome rather than a failure; populate `cited_sources` or you
should have refused.

*Judge* — four checks replacing one, with the rubric **enforced in code**:
`addresses_ticket`, `grounded_in_sources`, `touches_sensitive`, `cites_sources`.
SEND requires all three positives true and sensitive false. A model that reports
`grounded_in_sources: false` and then returns SEND has contradicted itself, so
the checks decide, not the verdict field.

*Judge code gates* — finally implementing round 0's decision. Two conditions are
facts about state rather than matters of opinion, so they run before any model
call: the responder explicitly refused, or `max_similarity < 0.40`. The floor
sits well below the 0.517-0.587 band on purpose. It claims only what the scores
support — "nothing retrieved is on this subject" — and leaves the
on-topic-but-unanswered case to the grounding check, which reads the text.

**Score: 5/7 — and the failures moved, which is the actual result.**

| Fixture | Round 1 | Round 2 |
|---|---|---|
| `refund-policy-question` | FAIL, invented a policy | **PASS** |
| `no-kb-answer` | FAIL, hallucinated availability | FAIL, *different cause* |
| `technical-in-kb` | PASS | FAIL, *new* |

`no-kb-answer` now behaves exactly as designed:

```
[responder] refused=True
[judge]     RETRY    -- responder declined
[responder] refused=True
[judge]     ESCALATE -- {'responder_refused': True}
outcome:    ESCALATE
```

It failed on `must_not_contain: ["minimum contract"]` because the refusal *names
what is missing*: "the knowledge base does not contain information about
self-hosted or on-premise deployment options, their costs, or minimum
contracts."

And `technical-in-kb` produced a correct, grounded, correctly-cited reply —
"the API rate limit is 600 requests per minute. When throttled, clients should
honor the Retry-After header and apply exponential backoff with jitter" — which
failed because the fixture demanded the literal string `"429"`.

**Both failures were the fixtures, not the system.**

---

## Round 3 — Fix the assertions

Two assertion-design bugs, both the same underlying error: **testing the wording
I imagined instead of the behaviour I required.**

1. `must_not_contain` now **exempts an explicit refusal**. Refusing to answer and
   asserting a falsehood are opposite behaviours and cannot share one substring
   check — a refusal necessarily echoes the question's vocabulary. A
   non-refusing draft is still checked, so a real hallucination is still caught;
   there are unit tests for both directions.

2. Added `must_contain_any` for assertions on substance rather than one token.
   The throttling half of `technical-in-kb` is answered correctly by naming the
   status code **or** the correct client behaviour.

This is the least glamorous round and the one I would bring up first. Rounds 1-2
improved the system; round 3 improved the *instrument*, after it scored a
correct escalation as a failure and a correct answer as incomplete. An eval that
is over-fitted to expected phrasing punishes correct behaviour it did not
anticipate, and the punishment looks exactly like a regression.

**Score: 7/7.** All fixtures measured, no infrastructure errors.

---

## Appendix — Five rounds where the harness measured itself

Every obstacle before round 1 was infrastructure wearing a result's clothing.
Recording them because the pattern is the most transferable thing here.

| # | Reported | Actually was |
|---|---|---|
| 1 | 6 prompt failures | DNS dropped mid-run; network errors bypassed retry entirely |
| 2 | (2h05m hang) | Retry amplification: 5 attempts x 3 models x 5 calls x 6 fixtures |
| 3 | 5/6 passing | Classifier short-circuited 4 fixtures past the retriever; assertions checked outcome, never mechanism |
| 4 | "every model failed" | 429 read as 503, tripping circuit breakers on healthy models — the suite disabled its own fallbacks |
| 5 | **"3/3 passed"** | 4 of 7 fixtures never ran; `RESULTS` was appended only after assertions, so errors left the denominator |

Round 4 also corrected a factual error in my own reasoning. I asserted quota was
per-project and built logic on it: never walk the chain, never trip the circuit,
just pace slower. The `quotaId` in the error payload said
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20 — **per day, per
model**. So each model has its own bucket (walking the chain is right), and
pacing tunes an axis a daily cap does not have.

What the harness grew in response, each item paid for by a specific wrong number:

- **Preflight check** — one cheap call before the suite; refuses to emit a score
  at all if the API is unreachable
- **`terminated_by` / `must_reach_retriever`** — assert the mechanism, so a right
  answer from the wrong node fails
- **Three-way failure taxonomy** — capacity (next model), quota (wait, or bench
  the model for the day), network (fail fast, say it is connectivity)
- **Denominator = fixtures declared**, never fixtures survived, with unmeasured
  fixtures called out as infrastructure rather than folded into the score
- **Stub keyed by role**, not by prompt substring, so rewriting a prompt cannot
  break three routing tests

The lesson is not "write assertions." It is that an eval suite spends much of its
life reporting numbers that are artifacts of its own plumbing, and the real work
is building enough discrimination into the harness that a plumbing result cannot
be mistaken for a measurement.

## Stability — is 7/7 repeatable, or was it one lucky sample?

A single passing run is one observation of a nondeterministic system, and
"it passes" and "it passed once" are different claims. Three consecutive runs:

| Run | Score | Runtime |
|---|---|---|
| 1 | 7/7 | 2m12s |
| 2 | 7/7 | 1m06s |
| 3 | 7/7 | 1m08s |

**3/3 runs at 7/7**, no assertion failures and no infrastructure errors. The
runtime drop after run 1 is capacity, not the pipeline -- fewer 503s meant
fewer fallback hops.

Three samples establishes repeatability, not a stability guarantee. The honest
claim is "it passed three consecutive runs," and the fixture most likely to
wobble is `ambiguous-two-issues`, whose `any_of` assertion accepts two
different correct handlings precisely because the input is genuinely ambiguous.

## Round 4 — Cross-provider benchmark, and the three bugs an ablation found

Adding the Groq backend made the same 7 fixtures runnable against a second
provider with the prompts held constant. Both passed 7/7, which is reassuring
and uninformative: when everything passes you learn nothing about where the
margin is.

So instead of comparing providers, I tested a **design claim**. I had asserted
the judge is the capability-critical role and gave it the strongest model on
that basis. That is falsifiable: downgrade one role at a time to the weakest
model and see which breaks.

| Configuration | Score |
|---|---|
| groq, full chain | 7/7 |
| groq, **judge** -> `gpt-oss-20b` | 7/7 |
| groq, **responder** -> `gpt-oss-20b` | **6/7** |

The responder mattered and the judge did not — the opposite of my assumption.
But reading the trace of the one failure showed the result was not a clean
capability finding at all. It was **three separate bugs**, two of them mine.

### Bug 1 — the judge could un-decide, and did

```
[judge] RETRY -- "Ticket involves a billing inquiry about seat charges,
                  which is a sensitive matter (failed: touches_sensitive)"
[responder] redrafted, shorter
[judge] SEND -- {'touches_sensitive': False, ...}
```

The judge flagged the ticket sensitive, the responder reworded, and the judge
then reported it **not** sensitive and approved the reply. On the *strong* model.

The defect is mine, in how I grouped the checks. `addresses_ticket`,
`grounded_in_sources` and `cites_sources` are faults in the **draft**, and a
redraft can genuinely fix them. `touches_sensitive` is a fact about the
**ticket** — a seat-billing dispute does not stop being about money because the
reply was reworded. By lumping it in with the others I handed the judge a second
independent chance to reach the opposite conclusion, converting one correct
escalation into a coin flip.

**Fix:** `touches_sensitive` is decided once and never retried. Only
draft-quality failures are redraftable. Two unit tests pin both halves.

### Bug 2 — my knowledge base contradicted itself

`account-team-members.md` said removing a member "frees their seat
immediately". `billing-plan-changes.md` said "removing seats takes effect at
the next cycle." Both were mine, and the responder cited whichever one
retrieval happened to surface — which was the more customer-favourable and
probably wrong answer.

Realistic, as knowledge-base bugs go, and invisible until a fixture happened to
ask a question spanning both. **Fix:** the access/billing distinction is now
explicit in one place and defers to the other document.

### Bug 3 — the fixture asserted something unretrievable

`ambiguous-two-issues` required citing `billing-plan-changes.md`. Retrieval
never surfaces it for that ticket — the top 4 chunks are webhooks twice,
team-members, and deletion. The assertion was **unsatisfiable**, so the fixture
was failing drafts for not citing a document they were never shown.

A third instance of the round-3 lesson: assert the substance (a billing source
was used, and the reply discusses seats and billing cycles), not one document I
assumed retrieval would pick.

### After the fixes

| Configuration | Score | Runtime |
|---|---|---|
| groq, full chain | 7/7 | 33s |
| groq, judge -> `gpt-oss-20b` | 7/7 | 73s |
| groq, responder -> `gpt-oss-20b` | 7/7 | 80s |
| gemini, full chain | 7/7 | 33s |

The ablation no longer separates the roles, because the bugs it exposed are
gone. Its value was diagnostic, not comparative — and the end state says
something about the architecture: **the pipeline passes with the 20B model in
either reasoning seat.** The code gates (explicit refusal, the 0.40 grounding
floor) and the sticky-sensitive rule carry enough of the safety burden that
model capability is not the binding factor.

Which is the original design thesis, arrived at by a longer road than intended:
put *not-knowing* in code where it is inspectable, and ask the model only for
the judgment code cannot make.
