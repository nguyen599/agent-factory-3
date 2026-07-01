# OLMo3 Proof RL

This recipe trains OLMo3 with `agent-factory-3` DDIS using full-parameter FSDP2 training and SGLang rollout servers.

## Layout

- Train node: 8 GPUs running `run_train.sh`.
- Rollout nodes: 2 nodes, each running two TP=4 SGLang servers.
- Model path must be HF format, for example `.hf_converted_checkpoints/step1100-hf`.
- Dataset must contain `problem` or `question`. CSV, JSON, JSONL, and parquet are supported.

## Rollout Servers

Run on each rollout node:

```bash
export MODEL_PATH=/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf
export RUN_DIR=/tmp/olmo3_agent_factory

bash examples/olmo3_proof/serve_rollout.sh 0,1,2,3 30100
bash examples/olmo3_proof/serve_rollout.sh 4,5,6,7 30101
```

### OLMo3 Attention Sink Rollouts

Set `OLMO3_USE_SINK=1` to use the `olmo3_sink` implementation from `olmo3-sink.md`.
If `MODEL_PATH` is a stock OLMo3 checkpoint, the script first creates a cached
`olmo3_sink` checkpoint with `self_attn.sinks`, then creates a SGLang deploy view
that matches the author's Kaggle serving patch (`model_type=olmo3`,
`architectures=["Olmo3SinkForCausalLM"]`, hybrid-SWA config enabled).

```bash
export OLMO3_USE_SINK=1
export OLMO3_SINK_CACHE_DIR=/tmp/olmo3_agent_factory/olmo3_sink_cache
export OLMO3_SINK_SGLANG_PATCH_ROOT=/home/manhnv/Documents/temp/git/math-train/patch-olmo3-sink-infer
export OLMO3_SINK_PATCH_SGLANG=1
export ATTENTION_BACKEND=triton

bash examples/olmo3_proof/serve_rollout.sh 0,1,2,3 30100
```

The rollout script applies the author-provided pure-Python SGLang patches at
runtime: OLMo3 sink model loading, DFlash sink support, SWA eviction, decode
tuning, and GQA-packed extend. Use `OLMO3_SINK_SGLANG_PATCH_SET=all` only when
you also want the W4A8 humming patch. Optional DFlash serving is enabled by
setting `DFLASH_DRAFT_MODEL_PATH`; `DFLASH_DRAFT_QUANTIZATION=compressed-tensors`
matches the int4-MLP draft path from `submisson-34.py`.

## Training

Run on the train node:

```bash
export MODEL_PATH=/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf
export DATASET_PATH=/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv
export RUN_DIR=/tmp/olmo3_agent_factory
export SGLANG_URLS=http://node1:30100,http://node1:30101,http://node2:30100,http://node2:30101

export BATCH_SIZE=32
export GROUP_SIZE=8
export MAX_BATCHES=100
export ROLLOUT_MAX_RESPONSE_TOKENS=60000
export ROLLOUT_MAX_CONTEXT_TOKENS=65536
export VERIFY_N=1
export META_N=0
export FORMAT_PARTIAL_SCORE=0.7
export PROOF_WEIGHT=0.76
export SELF_EVAL_WEIGHT=0.24
export WEIGHT_SYNC_INTERVAL=10
export MAX_STALENESS=10

bash examples/olmo3_proof/run_train.sh
```

For sink training, use the same sink cache and enable the sink loader:

```bash
export OLMO3_USE_SINK=1
export OLMO3_SINK_CACHE_DIR=/tmp/olmo3_agent_factory/olmo3_sink_cache
export OLMO3_SINK_ATTN_IMPLEMENTATION=olmo3_sink_fa3
```

If patched FA3 is unavailable, set `ATTN_IMPLEMENTATION=eager` for correctness
smoke tests only. Eager attention is not expected to be fast enough for 32B
long-context training.

To prepare paths manually:

```bash
PYTHONPATH=src python -m agent_factory_3.models.olmo3.sink_support prepare-model \
  --model-path "$MODEL_PATH" --cache-dir "$OLMO3_SINK_CACHE_DIR"

PYTHONPATH=src python -m agent_factory_3.models.olmo3.sink_support prepare-sglang \
  --model-path "$MODEL_PATH" --cache-dir "$OLMO3_SINK_CACHE_DIR"
```

## DeepSeekMath-V2 Reward

This example is for proof RL, not answer-only RLVR. Each rollout prompt asks the model to produce:

```text
## Solution
...

## Self Evaluation
Here is my evaluation of the solution:
...
Based on my evaluation, the final overall score should be:
\boxed{0|0.5|1}
```

The reward parser strips hidden thinking, extracts only the `## Solution` proof for proof judging, and extracts only the model's `## Self Evaluation` for meta judging. It does not send the full 60k-token reasoning trace to the judge.

Format handling:

- Missing or empty `## Solution` is fatal and gets reward `0`.
- A proof with `## Solution` but no self-evaluation or boxed self-score still goes to the verifier with `FORMAT_PARTIAL_SCORE`, default `0.7`.
- A fully formatted response gets format score `1.0`.

With `META_N=0`, reward is proof-score-only:

```text
reward = format_score * mean(verifier_scores)
```

With `META_N>0`, reward follows the DeepSeekMath-V2 generator formula:

```text
proof_score = mean(verifier_scores)
self_eval_score = mean(meta_verifier_scores)
score_alignment = 1 - abs(model_self_score - proof_score)
reward = format_score * (0.76 * proof_score + 0.24 * score_alignment * self_eval_score)
```

Set `VERIFY_N` for multiple proof-verifier samples and `META_N` for multiple meta-verifier samples. Prompt and response previews are written under `$RUN_DIR/prompt_previews` by default; override with `PROOF_REWARD_PREVIEW_DIR` or cap file size with `PROOF_REWARD_PREVIEW_MAX_CHARS`.
