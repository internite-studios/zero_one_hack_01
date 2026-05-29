# work/coach — Conversion Coach (UNIQA Insurance AI Track)

This directory will contain the Conversion Coach prototype for the UNIQA health insurance track.

## What belongs here

- **Journey state machine** — simulates the UNIQA calculator's 15 steps
- **Persona bots** — Judith (S1), Franz (S2), Peter (S3) as runnable agents
- **Detection layer** — behavioral signal detection (dwell time, back-nav, hover)
- **Decision layer** — intervention logic (when and how to intervene)
- **Simulation runner** — runs personas through journeys, measures before/after conversion
- **Results** — conversion rates, drop-off reductions, per-persona breakdowns

## Architecture (from track spec)

```
Insurance Chatbot (existing, handles domain Q&A)
       │
Conversion Coach (YOU BUILD THIS)
  ├── Detection layer: When to intervene?
  └── Decision layer: How to intervene?
       │
Persona Bots × 3 (YOU BUILD THESE)
  Judith / Franz / Peter — synthetic users with different intentions
```

## Scope boundary (critical)

| In scope | Out of scope |
|---|---|
| Private-doctor tariffs (Start, Optimal) | Hospital tariffs |
| "Myself only" path | "Other persons" path |
| Online-purchasable only | Opt. Plus & Premium (advisor-only) |

Conversion = online purchase of Start or Optimal tariff. Advisor handoffs are valid exits but don't count as conversions.

## Key numbers

- Baseline conversion: **5.6%** (1,000 starters → ~56 completions)
- 66% drop-off at initial price display
- 78% drop-off at final price
- Segment split: S1 30% / S2 50% / S3 20%

## Starting point

1. Read `tracks/insurance-uniqa/README.md`
2. Read `tracks/insurance-uniqa/Track_AI_Guided_Conversion_Flow_EN.md`
3. Read `tracks/insurance-uniqa/uniqa-funnel-doc_en.md`
4. Study the personas in `tracks/insurance-uniqa/personas_comparison_matrix.md`
5. Walk the live calculator: https://www.uniqa.at/rechner/krankenversicherung/

## Running simulations on Leonardo

Once the coach logic is built, run large-scale persona simulations:

```bash
# Sync to Leonardo
.\work\leonardo\ssh-leonardo.ps1 -sync

# SSH in
.\work\leonardo\ssh-leonardo.ps1

# Submit simulation job
sbatch work/leonardo/job_1gpu.slurm python work/coach/simulate.py --personas 1000 --runs 10
```
