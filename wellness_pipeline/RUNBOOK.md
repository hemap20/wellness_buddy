# Wellness Pipeline — Command Runbook

All commands assume you're in `wellness_pipeline/` with the venv active.

```bash
cd /Users/hemapunyamoorty/Desktop/chat_bot/wellness_pipeline
source .venv/bin/activate
export GEMINI_API_KEY="your-key"   # required for simulate/score — get one at https://aistudio.google.com/apikey
```

To use Claude instead of Gemini for simulator/judge: set `provider: "anthropic"` on `SimulatorConfig`/`JudgeConfig` in `config.py` and `export ANTHROPIC_API_KEY=...` instead.

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

cat checkpoints/dialogpt-small_phase2/train_summary.json
cat checkpoints/dialogpt-small_phase3/train_summary.json
```

Checkpoints (the actual weights) stay at `checkpoints/dialogpt-small_phase{2,3}/{step_N_pct_M,final}` — unchanged location. `run-phase` (step 6) copies the step numbers/paths and the loss curve from here into `results/` as metadata; the weights themselves are never duplicated there.

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
python3 orchestrator.py manifest                                   # every run, as a table
cat results/dialogpt-small/phase1_baseline.json                    # baseline: transcripts + judge_runs, all test cases/conditions in one file
cat results/dialogpt-small/n050/phase2/scores.json                 # phase 2: every judge run for every test case/condition
cat results/dialogpt-small/n050/phase2/checkpoints_meta.json       # step numbers + paths (not weights)
cat results/dialogpt-small/n050/phase2/loss_curve.json
ls results/dialogpt-small/n050/phase2/transcripts/
```

## 7. Aggregate report (reads the manifest, not the directory tree)

```bash
python3 report.py
cat reports/report_detail.csv     # one row per model × dataset_size × phase × condition × test_case
cat reports/report_summary.csv    # aggregated per model × dataset_size × phase × condition
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

`MODELS_UNDER_TEST` in `config.py` is the list — currently `dialogpt-small`, `gemma-3-1b-it` (`google/gemma-3-1b-it`), and `gemma-4-e2b-it` (`google/gemma-4-E2B-it`). Every stage already takes `--model <name>`, so adding a model to that list is enough to run the *entire* pipeline against it with the exact same commands, just naming it:

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

- **Sample efficiency isn't automated yet.** `train.py` saves checkpoints at 25/50/75/100% of steps (paths recorded in `checkpoints_meta.json`), but nothing loops over them to simulate+score each one automatically — `run-phase` only ever points at `final`. To get the efficiency curve, manually run step 4 against each `checkpoints/.../step_N_pct_M` path, then feed the scores into `metrics.py efficiency` against the matching entry in `phase1_baseline.json`.
- **`system_prompt_dependency` only resolves once sibling conditions exist on disk.** `run-phase` looks for sibling transcripts under that phase's `transcripts/` folder as it goes — score a transcript before its siblings (matched/no_prompt/paraphrased for the same test case) exist and you'll get `"N/A"` for that dimension. Running the whole phase in one `run-phase` call (as above) avoids this.
- **API volume**: `num_turns: 15` per test case × up to 3 conditions × 5 test cases × 3 judge runs per transcript adds up fast in Gemini calls. Do step 4 (single manual transcript) before a full `run-phase` to confirm your key/quota are good.
- **`--dataset-size` is not auto-detected from the data files** — it defaults to `config.py`'s `DataConfig.num_samples`, which can drift from what you actually passed to `data_prep.py --num-samples`. Double check they match before a `run-phase` call, or the results folder will be mislabeled.
- **`gemma-3-1b-it`'s `lora_target_modules` and chat-template behavior are unverified** — the repo is gated and couldn't be loaded in the environment this pipeline was built in. `model_utils.py` has a generic fallback if its chat template rejects a system-role turn (merges system content into the first user message and retries), but the actual template behavior hasn't been observed live. If the first real training run throws a LoRA target-module error, see the "Adding a third model" note in section 9.


{
  "count": 150,
  "avg": 1.5185309327666665,
  "p95": 2.419903268749999,
  "p99": 2.826060713249994,
  "transcripts_scanned": 10
}

{
  "checkpoint_score": 3.2,
  "baseline_score": 1.8,
  "steps_or_examples": 20,
  "delta": 1.4000000000000001,
  "sample_efficiency": 0.07
}