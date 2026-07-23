# Wellness Pipeline — Command Runbook

All commands assume you're in `wellness_pipeline/` with the venv active.

```bash
cd /Users/hemapunyamoorty/Desktop/chat_bot/wellness_pipeline
source .venv/bin/activate
# GEMINI_API_KEY and HF_TOKEN live in ../.env (chat_bot/.env), not this directory:
export $(grep -E "^(GEMINI_API_KEY|HF_TOKEN)" ../.env | xargs)
```

To use Claude instead of Gemini for simulator/judge: set `provider: "anthropic"` on `SimulatorConfig`/`JudgeConfig` in `config.py` and `export ANTHROPIC_API_KEY=...` instead.

---

## v3 pipeline (current generation — start here)

`results/`, `checkpoints/`, `logs/`, `charts/`, and `reports/` are now split into
`v1/`/`v2/`/`v3/` subtrees (see `results_manager.py::get_results_path` /
`get_version_root`) — each pipeline generation lives in its own directory,
per-model dirnames suffixed `_{version}` (e.g. `dialogpt-small_v3`), so
different generations never collide or overwrite each other on disk. `v1`
and `v2` are past generations (migrated, not written to anymore — see
`migrate_to_versioned_dirs.py`); `v3` is current (`config.PIPELINE_VERSION`).

**v3 adds, relative to v2:**
- **Per-model-tier LoRA rank/alpha** (`config.RANK_TIERS`/`rank_alpha_for()`) —
  replaces the single shared `lora_r=8`/`lora_alpha=16`. small (<350M
  params) → r=8/α=16, mid (1-4B) → r=16/α=32, large (7-8B) → r=32/α=64.
- **Real batching** (`config.BATCH_CONFIG`/`batch_settings_for()`) — training
  previously processed one example at a time despite having a
  `per_device_train_batch_size` field; it's now actually used, with padding/
  attention-masking via `train.py::collate_batch`, targeting an identical
  effective batch size of 8 across every model.
- **Preflight gates** (`preflight.py`) that BLOCK a model from training
  rather than just warning: chat-template validation (prints the formatted
  output for manual review — automated special-token check alone is not
  sufficient, see `config.EXPECTED_SPECIAL_TOKENS`), the target_modules/
  rank/trainable-% gate (reuses `lora_target_module_diagnostic.py`), and a
  sequence-length audit against `TrainingConfig.max_seq_length` (2048).
- **Checkpoint-based eval + best-checkpoint selection** (`checkpoint_eval.py`)
  — runs the existing simulator+judge eval suite against each of the 4
  checkpoint fractions (not just `final`), and picks an earlier checkpoint
  over the final one if `boundary_holding`/`crisis_handling` regressed more
  than `config.SAFETY_REGRESSION_THRESHOLD` (1.0pt) from its own max.

**The one command** — runs preflight → train (phase2→phase3) →
checkpoint-eval → charts/report, for every model that passes preflight:

```bash
python3 orchestrator.py run-v3 --dataset-size 50 --reviewed-templates
# --models dialogpt-small,qwen3-8b   to scope to specific models
# --reviewed-templates confirms you've read every printed chat-template
#   output — omit it and you'll get an interactive y/N prompt instead (or,
#   non-interactively, every model gets BLOCKED until you pass the flag)
```

A model that fails a gate is marked BLOCKED (reason printed) and skipped —
the rest of the batch still runs. `gemma-3-1b-it` (gated, unverified access)
is expected to BLOCKED at the load step unless you've completed its HF
access setup (see section 9 below).

Diagnose target_modules/rank issues for one model standalone (same logic
`preflight.py` calls into):
```bash
python3 lora_target_module_diagnostic.py --models dialogpt-small
```

Every existing command below (`train.py`, `orchestrator.py train/run-phase/
run-all/run-batch/run-full`, `report.py`, `generate_charts.py`) now also
accepts `--version` (defaults to `config.PIPELINE_VERSION`, currently `v3`)
if you need to target a specific generation instead of the current one —
e.g. `python3 report.py --version v2` to regenerate v2's report from its
manifest.

---

## 0. Sanity checks (no LLM calls, cheap)

```bash
python3 -m py_compile config.py data_prep.py model_utils.py train.py inference.py simulator.py judge.py metrics.py report.py orchestrator.py llm_client.py results_manager.py
python3 results_manager.py   # standalone demo of the results/manifest logic, runs in an isolated temp dir
```

---

## 1. Data prep

```bash
python3 data_prep.py --num-samples 50
cat data/phase2_pairs.jsonl   # no system role
cat data/phase3_pairs.jsonl   # fixed system prompt prepended
```

`DataConfig.num_samples` in `config.py` is the canonical dataset-size value — if you pass `--num-samples` here, pass the **same number** to `run-phase --dataset-size` in step 6, or the results folder name (`n{size}`) won't match what you actually trained on.

## 2. Training (LoRA, both phases)

```bash
python3 train.py --phase 2 --num-samples 50
python3 train.py --phase 3 --num-samples 50

cat checkpoints/v3/dialogpt-small_v3_n050_phase2/train_summary.json
cat checkpoints/v3/dialogpt-small_v3_n050_phase3/train_summary.json
```

Checkpoints (the actual weights) live at `checkpoints/{version}/{model}_{version}_n{size}_phase{N}/{step_N_pct_M,final}` — namespaced by pipeline version + dataset size (same `n050`/`n500` convention as `data/` and `results/`), so re-training the same model/phase at a different `--num-samples` or a different pipeline generation never overwrites a previous run's weights. `run-phase` (step 6) copies the step numbers/paths and the loss curve from here into `results/` as metadata; the weights themselves are never duplicated there. `train_summary.json` also now records `rank_tier`/`lora_r`/`lora_alpha`/`per_device_batch_size`/`gradient_accumulation_steps`/`effective_batch_size` for that run — see the v3 section above.

> **Historical note:** `v1`/`v2`-generation checkpoint dirs (some of which predate dataset-size namespacing entirely, e.g. `dialogpt-small_phase2` with no `_n050_` segment) were migrated into `checkpoints/v1/`/`checkpoints/v2/` by `migrate_to_versioned_dirs.py` — see them there, not at the top level, and don't expect them to follow the current naming convention.

## 3. Inference smoke test (no simulator/judge, just one generation)

```bash
python3 inference.py --model dialogpt-small
python3 inference.py --model dialogpt-small \
  --adapter checkpoints/dialogpt-small_phase3/final \
  --system-prompt "You are a warm, supportive wellness companion. Keep responses casual and conversational." \
  --user-message "I've had a really rough day."
```

## 4. One manual transcript + score (do this before a full phase run)

This uses the **old flat `transcripts/`/`scores/` dirs**, kept specifically for quick one-off debugging outside the `results/` structure — good for confirming your API key/quota work before spending calls on a full phase.

```bash
python3 simulator.py --test-case T1 --phase phase2 --condition no_prompt \
  --adapter checkpoints/dialogpt-small_phase2/final
# --mode defaults to adaptive. Only pass --mode fixed if turn_beats are literal
# quoted messages you wrote yourself — otherwise the bot ends up replying to
# stage-direction text instead of natural dialogue.

cat transcripts/dialogpt-small_phase2_no_prompt_T1_adaptive.json   # eyeball it

python3 judge.py --transcript transcripts/dialogpt-small_phase2_no_prompt_T1_adaptive.json
cat scores/dialogpt-small_phase2_no_prompt_T1_adaptive.scores.json
```

## 5. Full phase runs — the real experiment record

This is what actually populates `results/{model}/n{size}/phase{N}/` and `results/manifest.jsonl`, via `results_manager.py`. Each call: simulates every (test_case × condition) combo for that phase, scores every transcript 3x (all runs kept, not averaged down), copies training metadata, appends one manifest line per condition, then runs the report once at the end.

```bash
# Phase 1 — untrained baseline, per-model only (NOT per dataset size).
# Skips automatically if results/{model}/phase1_baseline.json already exists.
python3 orchestrator.py run-phase --phase 1

# Phase 2 — pairs-only fine-tune. --dataset-size must match what you actually
# trained on (defaults to config.py's DataConfig.num_samples if omitted).
python3 orchestrator.py run-phase --phase 2 \
  --checkpoint checkpoints/dialogpt-small_phase2/final --dataset-size 50

# Phase 3 — pairs + fixed system prompt, jointly trained
python3 orchestrator.py run-phase --phase 3 \
  --checkpoint checkpoints/dialogpt-small_phase3/final --dataset-size 50
```

Re-running the same (model, dataset_size, phase) combo **raises an error instead of overwriting** unless you pass `--force`:

```bash
python3 orchestrator.py run-phase --phase 2 --checkpoint checkpoints/dialogpt-small_phase2/final --dataset-size 50 --force
```

`--mode` defaults to `adaptive`; pass `--both-modes` to also generate the literal-script `fixed` variant once your turn_beats are written as real quoted messages.

## 6. Inspect results

```bash
python3 orchestrator.py manifest                                   # every run in the current version, as a table
cat results/v3/dialogpt-small_v3/phase1_baseline.json               # baseline: transcripts + judge_runs, all test cases/conditions in one file
cat results/v3/dialogpt-small_v3/n050/phase2/scores.json            # phase 2: every judge run for every test case/condition
cat results/v3/dialogpt-small_v3/n050/phase2/checkpoints_meta.json  # step numbers + paths (not weights)
cat results/v3/dialogpt-small_v3/n050/phase2/loss_curve.json
ls results/v3/dialogpt-small_v3/n050/phase2/transcripts/
```

## 7. Aggregate report (reads the manifest, not the directory tree)

```bash
python3 report.py
cat reports/v3/report_detail.csv     # one row per model × dataset_size × phase × condition × test_case
cat reports/v3/report_summary.csv    # aggregated per model × dataset_size × phase × condition
```

`report.py` discovers everything via `results/manifest.jsonl`. For the old flat-dir debug runs from step 4 instead: `python3 report.py --legacy-scores-dir scores`.

## 8. Automatic (non-judge) metrics

```bash
# per-phase training loss curve (same data as results/.../loss_curve.json, read from the original log)
python3 metrics.py loss --loss-log logs/dialogpt-small_phase2_loss.jsonl

# latency across a set of transcripts
python3 metrics.py latency --transcripts "results/dialogpt-small/n050/phase2/transcripts/*.json"

# sample efficiency (needs a checkpoint's judge score + the phase-1 baseline score,
# computed by hand from report_detail.csv until the checkpoint-sweep is automated)
python3 metrics.py efficiency --checkpoint-score 3.2 --baseline-score 1.8 --steps 15
```

---

## Cleanup / re-running after a bug fix

```bash
# stale flat-dir debug transcripts (step 4) affected by the fixed-vs-adaptive mode bug
rm transcripts/*_fixed.json scores/*_fixed.scores.json

# to force-redo a results/ run for the same model+size+phase
python3 orchestrator.py run-phase --phase 2 --checkpoint ... --dataset-size 50 --force

# to regenerate an existing baseline (rare — baseline is meant to be stable)
python3 orchestrator.py run-phase --phase 1 --force
```

---

## 9. Testing a second (or third) model

`MODELS_UNDER_TEST` in `config.py` is the list — currently 8 entries: `dialogpt-small`, `gemma-3-1b-it`, `gemma-4-e2b-it`, `starling-lm-7b-alpha`, `empathetic-qwen3-8b-jan`, `qwen3-8b`, `mistral-7b-instruct-v0.3`, `llama-3-8b`. `gemma-3-1b-it` is currently BLOCKED — gated repo, access not yet approved for the configured `HF_TOKEN` (confirmed by `preflight.py`/`lora_target_module_diagnostic.py`: it fails to load even with the token exported, unlike `mistral-7b-instruct-v0.3` and `llama-3-8b` which load fine with the same token). The other 7 are active/trained under v3. Every stage already takes `--model <name>`, so adding a model to that list is enough to run the *entire* pipeline against it with the exact same commands, just naming it:

```bash
# gemma-3-1b-it is a GATED repo — one-time setup before anything else works:
#   1. accept the license at https://huggingface.co/google/gemma-3-1b-it
#   2. huggingface-cli login   (or: export HF_TOKEN=hf_...)

python3 train.py --phase 2 --model gemma-3-1b-it --num-samples 50
python3 train.py --phase 3 --model gemma-3-1b-it --num-samples 50

python3 inference.py --model gemma-3-1b-it
python3 simulator.py --test-case T1 --phase phase2 --condition no_prompt \
  --model gemma-3-1b-it --adapter checkpoints/gemma-3-1b-it_phase2/final

python3 orchestrator.py run-phase --phase 1 --model gemma-3-1b-it
python3 orchestrator.py run-phase --phase 2 --model gemma-3-1b-it \
  --checkpoint checkpoints/gemma-3-1b-it_phase2/final --dataset-size 50
```

Results land at `results/gemma-3-1b-it/...`, fully separate from `results/dialogpt-small/...` — `report.py` will show both once both have runs, one row per model as always.

**What's genuinely shared vs. what had to be per-model:** `TrainingConfig` (LoRA rank/alpha/dropout, learning rate, epochs, batch size, max_seq_length, checkpoint fractions) is identical across every model, per the original design. `lora_target_modules` moved onto `ModelUnderTest` instead — it's naming the actual attention layers to adapt (`c_attn` for GPT-2, `q_proj`/`k_proj`/`v_proj`/`o_proj` for Gemma/Llama-style models), which is architecture, not a tunable hyperparameter; every other setting is still literally the same `SHARED_TRAINING_CONFIG` object for both models.

**Adding a third model:** append another `ModelUnderTest(...)` entry in `config.py`. You'll need its real `lora_target_modules` — inspect with:
```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("the/model-id")
for n, _ in m.named_modules():
    print(n)
```
and pick the attention projection layer names (commonly `q_proj`/`k_proj`/`v_proj`/`o_proj`, or `c_attn` for GPT-2-family). Gemma-3-1b-it's target modules here were set from public architecture docs, **not verified against the live model** (it's gated, couldn't be downloaded in the environment this was built in) — if training throws a "target module not found" error, that's the thing to fix.

`gemma-4-e2b-it` is different — it's **not gated**, so `lora_target_modules` was verified directly against the installed `transformers` source (`Gemma4TextAttention`, plain `nn.Linear` q/k/v/o_proj) rather than guessed.

## 10. Thinking mode (`gemma-4-e2b-it`)

`gemma-4-e2b-it`'s chat template has a real `enable_thinking` toggle (`ModelUnderTest.supports_thinking=True`). When a model has this flag set, `orchestrator.py run-phase` automatically runs **every** test_case × condition **twice** — once with thinking on, once off — no extra flag needed:

```bash
python3 orchestrator.py run-phase --phase 1 --model gemma-4-e2b-it
python3 orchestrator.py run-phase --phase 2 --model gemma-4-e2b-it \
  --checkpoint checkpoints/gemma-4-e2b-it_phase2/final --dataset-size 50
```

This doubles the Gemini call volume for that model (twice as many transcripts to simulate+score) — keep that in mind before a big `--dataset-size`.

Transcripts get a `_thinking_on`/`_thinking_off` suffix (`results/gemma-4-e2b-it/n050/phase2/transcripts/T1_no_prompt_adaptive_thinking_on.json`), manifest entries and `report.py` rows carry a `thinking_mode` column (`true`/`false`/`null` — null for models without the toggle), and `system_prompt_dependency` sibling-matching only compares transcripts within the *same* thinking mode (so that dimension stays about prompt-condition consistency, not conflated with the thinking toggle).

For a single manual check outside `run-phase`:
```bash
python3 inference.py --model gemma-4-e2b-it --thinking on --user-message "I've had a really rough day."
python3 inference.py --model gemma-4-e2b-it --thinking off --user-message "I've had a really rough day."
```

The model's thinking segment (delimited by special tokens in its raw output) is automatically stripped from the returned response — you'll only ever see the final answer in transcripts/inference output, never the raw chain-of-thought, so thinking-on and thinking-off transcripts are directly comparable.

---

## Known gaps / things to know before scaling up

- **Sample efficiency across checkpoints is now automated for v3** via `checkpoint_eval.py` (called automatically by `orchestrator.py run-v3`) — it runs the eval suite against each of the 4 checkpoint fractions (all 8 test cases, 1 condition, `judge_cfg.num_runs` overridden to `config.CHECKPOINT_EVAL_JUDGE_RUNS=1` to keep 4×7 evals affordable) and writes a trajectory report to `reports/v3/{model}_checkpoint_trajectory.md` + `.json`, including which checkpoint was selected as "best" and why. `metrics.py efficiency` is still there for a one-off manual computation, but you shouldn't need it for routine v3 runs anymore.
- **`system_prompt_dependency` only resolves once sibling conditions exist on disk.** `run-phase` looks for sibling transcripts under that phase's `transcripts/` folder as it goes — score a transcript before its siblings (matched/no_prompt/paraphrased for the same test case) exist and you'll get `"N/A"` for that dimension. Running the whole phase in one `run-phase` call (as above) avoids this. Checkpoint-eval's cheaper 1-condition passes always get `"N/A"` here — expected, not a bug (see checkpoint_eval.py's docstring).
- **API volume**: `num_turns: 20` per test case × up to 3 conditions × 8 test cases × 3 judge runs per transcript adds up fast in Gemini calls, and `run-v3` multiplies that across every model in the batch (plus 4 cheaper checkpoint-eval passes each). Do step 4 (single manual transcript) before a full `run-phase`/`run-v3` to confirm your key/quota are good, and scope `run-v3 --models` to one model first (as we did for the initial v3 smoke test) before running the full 7-model batch.
- **`--dataset-size` is not auto-detected from the data files** — it defaults to `config.py`'s `DataConfig.num_samples`, which can drift from what you actually passed to `data_prep.py --num-samples`. Double check they match before a `run-phase`/`run-v3` call, or the results folder will be mislabeled.
- **`gemma-3-1b-it` is currently BLOCKED, confirmed via live diagnostic** (not just "unverified" as before) — `HF_TOKEN` is valid and works for `mistral-7b-instruct-v0.3`/`llama-3-8b`, but `google/gemma-3-1b-it` still returns a gated-repo error for this account. Needs the HF access request approved before it can be diagnosed or trained at all; `preflight.py`/`run-v3` will keep reporting it BLOCKED at the load step until then.
- **gemma-4-e2b-it's LoRA coverage is intentionally asymmetric** — its text decoder has cross-layer KV-sharing (`config.text_config.num_kv_shared_layers=20`): 20 of 35 decoder layers have no `k_proj`/`v_proj` modules at all (confirmed via `named_modules()` inspection), so LoRA only adapts `q_proj`/`o_proj` there vs. full q/k/v/o in the other 15 layers. This is captured in `config.KNOWN_ARCHITECTURAL_ASYMMETRIES` and surfaced as "expected-asymmetric" (not BLOCKED) by both `preflight.py` and `lora_target_module_diagnostic.py` — don't try to "fix" it by changing `target_modules` for this model.