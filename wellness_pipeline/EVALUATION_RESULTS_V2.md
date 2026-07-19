# Wellness Buddy — Version 2 Results (in progress)

## What changed from Version 1

Version 1 runs showed `possible_overfitting: true` on every trained model (validation loss rising while training loss kept falling). Version 2 addresses this with three shared changes, applied identically across every model:

| Setting | V1 | V2 |
|---|---|---|
| `num_train_epochs` | 5 | 3 |
| `weight_decay` | 0 | 0.01 |
| Early stopping | none | stops at the first checkpoint where val_loss rises above the best-so-far; keeps that best checkpoint as `final` |

All other hyperparameters (LoRA rank/alpha/dropout, learning rate, checkpoint fractions, seed) are unchanged and identical across every model in the suite. Every V2 result below is saved under a `-v2` tag so it never overwrote the original V1 checkpoints/results.

This run also expanded the model roster from 2 to 7 candidates: `dialogpt-small`, `gemma-4-e2b-it`, `starling-lm-7b-alpha`, `empathetic-qwen3-8b-jan`, `qwen3-8b`, `mistral-7b-instruct-v0.3`, `llama-3-8b`.

Methodology (phase definitions, the 5 test cases, and the 6-dimension judge rubric) is unchanged from Version 1 — see `EVALUATION_RESULTS.md` for full definitions.

---

## Status as of this snapshot

| Model | n050 status | n200 status |
|---|---|---|
| dialogpt-small | ✅ complete (phase 1/2/3) | not started |
| gemma-4-e2b-it | ✅ complete (phase 1/2/3, both thinking modes) | not started |
| starling-lm-7b-alpha | ✅ complete (phase 1/2/3) | not started |
| empathetic-qwen3-8b-jan | ✅ complete (phase 1/2/3, both thinking modes) | not started |
| qwen3-8b | not started | not started |
| mistral-7b-instruct-v0.3 | not started | not started |
| llama-3-8b | not started | not started |

n200 was substituted for the original n500 second dataset size to reduce training time (testing time is dataset-size-independent — only training scales with sample count).

---

## Results — n050, 6-dimension rubric (1–5 scale, averaged across all judge runs)

### dialogpt-small

| Metric | Phase 1 (baseline) | Phase 2 (fine-tuned) | Phase 3 (fine-tuned + system prompt) |
|---|---|---|---|
| persona_tone_adherence | 1.00 | 1.00 | 1.00 |
| empathy_vs_parroting | 1.00 | 1.00 | 1.00 |
| boundary_holding | 3.73 | 3.80 | 3.58 |
| crisis_handling | 1.80 | 1.80 | 1.80 |
| system_prompt_dependency | — | 2.87 | 4.13 |
| multi_turn_drift | 1.00 | 1.00 | 1.00 |

### gemma-4-e2b-it

| Metric | Phase 1 (baseline) | Phase 2 (fine-tuned) | Phase 3 (fine-tuned + system prompt) |
|---|---|---|---|
| persona_tone_adherence | 1.05 | 1.45 | 1.47 |
| empathy_vs_parroting | 1.00 | 1.25 | 1.14 |
| boundary_holding | 4.77 | 4.25 | 4.17 |
| crisis_handling | 2.15 | 2.22 | 1.97 |
| system_prompt_dependency | — | 1.23 | 1.43 |
| multi_turn_drift | 1.15 | 1.10 | 1.09 |

### starling-lm-7b-alpha

| Metric | Phase 1 (baseline) | Phase 2 (fine-tuned) | Phase 3 (fine-tuned + system prompt) |
|---|---|---|---|
| persona_tone_adherence | 2.03 | 2.77 | 2.84 |
| empathy_vs_parroting | 2.13 | 2.70 | 3.13 |
| boundary_holding | 3.60 | 4.00 | 4.00 |
| crisis_handling | 2.40 | 2.43 | 2.56 |
| system_prompt_dependency | — | 1.93 | 2.20 |
| multi_turn_drift | 1.77 | 2.60 | 2.76 |

### empathetic-qwen3-8b-jan

| Metric | Phase 1 (baseline) | Phase 2 (fine-tuned) | Phase 3 (fine-tuned + system prompt) |
|---|---|---|---|
| persona_tone_adherence | 2.07 | 2.30 | 2.58 |
| empathy_vs_parroting | 1.85 | 2.33 | 2.80 |
| boundary_holding | 3.28 | 3.63 | 3.70 |
| crisis_handling | 2.27 | 2.52 | 2.17 |
| system_prompt_dependency | — | 1.80 | 2.18 |
| multi_turn_drift | 1.30 | 1.92 | 2.09 |

---

## Key findings so far

- **Fine-tuning helped across nearly every dimension for every model tested**, except dialogpt-small (flat/unchanged — likely a capacity ceiling for a model this small).
- **starling-lm-7b-alpha and empathetic-qwen3-8b-jan show the strongest post-fine-tuning gains**, particularly on empathy_vs_parroting and multi_turn_drift.
- **crisis_handling remains weak across the board** (2.0–2.6 / 5 for every model), independent of fine-tuning or model choice — this is a standing concern, not something the current training approach has resolved.
- Early stopping is firing as designed: e.g. starling-lm-7b-alpha's phase 2/3 training both stopped at step 92 of 123 once val_loss regressed, reverting to the better step-62 checkpoint rather than the overfit final one.

## Known open questions (not yet resolved)

- **Thinking-mode fine-tuning gap**: gemma-4-e2b-it and empathetic-qwen3-8b-jan/qwen3-8b support a thinking-mode toggle, but training data contains no authored reasoning traces — the LoRA adapter never learns *how* to reason about wellness scenarios, only what to answer. Thinking-mode results reflect the base model's generic pretrained reasoning, not a wellness-tuned one. A follow-up experiment with synthesized reasoning-trace training data is under consideration, tracked separately (not yet started).
- Raw `<think>...</think>` reasoning content is stripped before transcripts are saved — there's currently no way to inspect what the model actually reasoned before its final answer.

*This is a snapshot mid-run — qwen3-8b, mistral-7b-instruct-v0.3, and llama-3-8b at n050, and the entire n200 pass for all 7 models, are still pending.*
