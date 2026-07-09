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
python3 -m py_compile config.py data_prep.py model_utils.py train.py inference.py simulator.py judge.py metrics.py report.py orchestrator.py llm_client.py
```

---

## 1. Data prep

```bash
python3 data_prep.py --num-samples 5
cat data/phase2_pairs.jsonl   # no system role
cat data/phase3_pairs.jsonl   # fixed system prompt prepended
```

## 2. Training (LoRA, both phases)

```bash
python3 train.py --phase 2
python3 train.py --phase 3

cat checkpoints/dialogpt-small_phase2/train_summary.json
cat checkpoints/dialogpt-small_phase3/train_summary.json
```

Checkpoints land at `checkpoints/dialogpt-small_phase{2,3}/{step_N_pct_M,final}`.

## 3. Inference smoke test (no simulator/judge, just one generation)

```bash
python3 inference.py --model dialogpt-small
python3 inference.py --model dialogpt-small \
  --adapter checkpoints/dialogpt-small_phase3/final \
  --system-prompt "You are a warm, supportive wellness companion. Keep responses casual and conversational." \
  --user-message "I've had a really rough day."
```

## 4. One manual transcript + score (do this before a full phase run)

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

## 5. Aggregate whatever's been scored so far

```bash
python3 report.py
cat reports/report_detail.csv
cat reports/report_summary.csv
```

## 6. Full phase runs (all 5 test cases × all conditions for that phase)

```bash
# Phase 1 — untrained baseline. Run this FIRST, it's what 2/3 get compared against.
python3 orchestrator.py run-phase --phase 1

# Phase 2 — pairs-only fine-tune
python3 orchestrator.py run-phase --phase 2 --checkpoint checkpoints/dialogpt-small_phase2/final

# Phase 3 — pairs + fixed system prompt, jointly trained
python3 orchestrator.py run-phase --phase 3 --checkpoint checkpoints/dialogpt-small_phase3/final
```

Each `run-phase` call: simulates every (test_case × condition) combo for that phase, scores every transcript 3x, then runs `report.py` once at the end. `--mode` defaults to `adaptive`; pass `--both-modes` to also generate the literal-script `fixed` variant once your turn_beats are written as real quoted messages.

## 7. Automatic (non-judge) metrics

```bash
# per-phase training loss curve
python3 metrics.py loss --loss-log logs/dialogpt-small_phase2_loss.jsonl
python3 metrics.py loss --loss-log logs/dialogpt-small_phase3_loss.jsonl

# latency across a set of transcripts
python3 metrics.py latency --transcripts "transcripts/*.json"

# sample efficiency (needs a checkpoint's judge score + the phase-1 baseline score,
# computed by hand from report_detail.csv until the checkpoint-sweep is automated)
python3 metrics.py efficiency --checkpoint-score 3.2 --baseline-score 1.8 --steps 15
```

---

## Cleanup (only if you need to re-run something that was affected by a bug)

```bash
rm transcripts/*_fixed.json scores/*_fixed.scores.json   # stale label-text-artifact runs
python3 report.py                                        # regenerate the aggregate after cleanup
```

---

## Known gaps / things to know before scaling up

- **Sample efficiency isn't automated yet.** `train.py` saves checkpoints at 25/50/75/100% of steps, but nothing loops over them to simulate+score each one automatically — `run-phase` only ever points at `final`. To get the efficiency curve, manually run step 4 above against each `checkpoints/.../step_N_pct_M` path, then feed the scores into `metrics.py efficiency`.
- **`system_prompt_dependency` only resolves once sibling conditions exist on disk.** Score a transcript before its siblings (matched/no_prompt/paraphrased for the same test case) exist and you'll get `"N/A"` for that dimension. Run all conditions for a phase (`run-phase` does this) before scoring, or re-score afterward with `judge.py` once all conditions are present.
- **API volume**: `max_turns: 15` per test case × up to 3 conditions × 5 test cases × 3 judge runs per transcript adds up fast in Gemini calls. Do step 4 (single manual transcript) before a full `run-phase` to confirm your key/quota are good.
