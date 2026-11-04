# Multi-Agent Last-Mile Delivery Exception Handling

A production-style, non-notebook implementation of a multi-agent system
that triages last-mile delivery exceptions (failed attempts, damaged
packages, weather delays, address issues, refusals), decides a resolution,
drafts the customer notification, and routes anything risky to a human --
built on **LangGraph**, with a deterministic rule engine, prompt-injection
and noise guardrails, PII-isolated agent views, and a five-metric
evaluation harness.

This is the engineering counterpart to a notebook-based coursework project
on the same problem: same dataset, same business logic, rebuilt as a
testable Python package instead of a sequence of notebook cells.

## Why this exists

Delivery exception handling is a good small-scale example of a problem
real agentic systems run into constantly: multiple specialized agents need
to collaborate, some of them should never see certain data, a
deterministic policy has to be enforced even when an LLM disagrees with
it, and the whole thing has to be auditable after the fact. This project
is a compact, inspectable implementation of all four of those concerns in
one place.

## What it does

For each delivery event, a 7-node LangGraph workflow:

1. **Deduplicates and consolidates** multiple scan rows into one event.
2. **Screens for prompt injection** in driver notes and retrieved policy
   text, and **filters routine noise** (a normal "delivered" scan) before
   spending a single LLM call on it.
3. Runs a **deterministic rule engine** for escalation triggers (3rd
   attempt, VIP with repeated exceptions, damaged perishables, long
   weather delays, fraud-pattern addresses) that an LLM cannot override.
4. Has a **Resolution Agent** classify the exception and pick an action
   (reschedule, reroute to locker, replace, return to sender), grounded in
   playbook excerpts retrieved from a vector store.
5. Has a **Critic** validate that decision against the playbook and either
   accept it, send it back for one revision, or escalate it.
6. Has a **Communication Agent** -- the *only* agent that can see the
   customer's name -- draft the customer-facing message.
7. Has a second **Critic** check that message for tone, accuracy and
   safety before it ships.

See [`docs/architecture.md`](docs/architecture.md) for the full graph
diagram and the PII-isolation design in detail.

## Project structure

```
multiagent-delivery-ops/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                  # copy to .env; never commit real keys
├── data/sample/                  # sample dataset this repo ships with
│   ├── customers.db               # SQLite: customers + lockers
│   ├── delivery_logs.csv          # raw scan events per shipment
│   ├── ground_truth.csv           # labeled exception/resolution/tone/escalation
│   └── exception_resolution_playbook.pdf
├── src/delivery_ops/
│   ├── config.py                  # env-driven configuration
│   ├── ingestion.py               # loads CSV/SQLite/PDF, builds the vector store
│   ├── tools.py                   # the 5 LangChain tools agents call
│   ├── guardrails.py              # injection scanner, noise filter, dedup/consolidate
│   ├── state.py                   # UnifiedAgentState + per-agent views
│   ├── schemas.py                 # Pydantic structured-output models
│   ├── prompts.py                 # every system prompt, in one place
│   ├── runtime.py                 # wires LLM clients + data + tools together
│   ├── graph.py                   # builds and compiles the LangGraph workflow
│   ├── evaluation.py              # the 5 metrics + single-shipment test runner
│   └── nodes/                     # one file per graph node
├── scripts/run_demo.py            # CLI: run every shipment, print a full report
├── tests/                         # offline unit tests (no LLM/network needed)
└── docs/architecture.md           # graph diagram + design notes
```

## Getting started

```bash
git clone https://github.com/harshal1900/multiagent-delivery-ops.git
cd multiagent-delivery-ops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # fill in OPENAI_API_KEY (and OPENAI_API_BASE if using a gateway)

python scripts/run_demo.py      # runs all 10 sample shipments end to end
```

No cloud setup is required to try the deterministic pieces -- the rule
engine, the guardrails, and the PII-isolation logic are all covered by
offline tests:

```bash
pytest tests/ -v
```

By default the pipeline reads the sample dataset shipped in
`data/sample/`. Point `DATA_DIR` in `.env` at a different folder with the
same four files to run it against another dataset.

## Sample dataset

`data/sample/` contains a small synthetic dataset (10 shipments, 12
customers, 6 pickup lockers, and a 10-page exception-resolution playbook
PDF) used throughout development and in the numbers below. It contains no
real customer data.

## Evaluation

`scripts/run_demo.py` computes five metrics per shipment and an aggregate
summary. On the sample dataset with `gpt-4o` as both the generation and
evaluation model, one evaluation run observed:

| Metric                        | Result        |
|--------------------------------|---------------|
| Task completion rate           | 90% (9/10)    |
| Exception classification       | 10/10         |
| Resolution action correct      | 9/10          |
| Communication tone correct     | 10/10         |
| Escalation accuracy            | 75% (6/8 evaluated) |
| Tool call accuracy             | 100% (10/10)  |
| Mean reasoning coherence       | 4.0 / 5       |
| Mean end-to-end latency        | ~7-8 s/shipment |

These numbers are illustrative of a 10-case sample, not a claim of
production accuracy -- LLM outputs vary run to run, and the two misses
observed are discussed below because they're more interesting than the
pass rate.

### What the misses actually tell you

- One shipment's ground-truth resolution (`REROUTE_TO_LOCKER`) was
  infeasible given the data: the only locker in that zip code was `FULL`,
  and the playbook forbids rerouting to a full locker. The system correctly
  fell back to `RETURN_TO_SENDER` -- a case where the labeled "correct"
  answer and the stated policy actually conflict.
- Two shipments show a Critic-driven escalation the deterministic rule
  engine did not call for. In one, the Critic used both revision loops
  before the orchestrator forced an escalation at the loop cap; in the
  other, the Critic escalated a message that had already matched the
  correct resolution and tone. Both are visible in the `trajectory_log`
  this project records for exactly this kind of audit, which is the
  argument for keeping that log in the first place.

### Suggested next steps

- Move the discretionary escalation call fully into the orchestrator's
  deterministic logic rather than leaving it to Critic judgment, to reduce
  variance run to run.
- Have the Critics log a structured reason code rather than free text, so
  patterns like "double REVISE then forced escalate" are queryable instead
  of only readable.
- Scale the evaluation set well past 10 curated cases before treating any
  of these numbers as representative.

## Design notes worth skimming

- **The orchestrator never calls an LLM.** It is the single place that
  enforces the automatic escalation triggers and the revision-loop cap, so
  a disagreement between two LLM agents can never become an infinite loop
  or a silently dropped escalation.
- **PII isolation is structural, not a prompt instruction.** The
  Communication Agent is the only node whose view (`CommunicationAgentView`
  in `state.py`) includes the customer's name. Every other agent is
  physically handed a dict that never had the field in it.
- **Guardrails run before the first LLM call, not after.** Prompt
  injection in driver notes or retrieved playbook text, and routine noise
  with no anomaly keywords, are both caught in the preprocessor -- so a
  malicious or routine input never costs a model call, let alone gets a
  chance to steer one.

## License

MIT -- see [LICENSE](LICENSE).
