"""OLMo3 proof RL recipe using agent-factory-3 DDIS.

This recipe expects SGLang rollout servers to already be running. It uses
full-parameter FSDP2 training and periodically syncs merged weights back to
the SGLang servers.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from agent_factory_3.models.olmo3 import load_olmo3_causal_lm
from agent_factory_3.models.olmo3.sink_support import prepare_olmo3_sink_model_path
from agent_factory_3.orchestrator.types import per_rollout
from agent_factory_3.rewards import OLMoProofReward, OLMoProofRewardConfig, build_deepseek_proof_generation_prompt
from agent_factory_3.rollout import ConversationConfig, LoopConfig, RecordConfig, SamplingParams
from agent_factory_3.trainer.model_trainer import ModelTrainer
from agent_factory_3.trainer.rl_flow import RLFlow
from agent_factory_3.trainer.training_config import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf",
)
DATASET_PATH = os.environ.get("DATASET_PATH", "/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv")
RUN_DIR = os.environ.get("RUN_DIR", "./runs/olmo3_proof")
SGLANG_URLS = env_list("SGLANG_URLS", "http://127.0.0.1:30100")
JUDGE_URLS = env_list("JUDGE_URLS") or SGLANG_URLS


def resolve_model_path() -> str:
    if not env_bool("OLMO3_USE_SINK", False):
        return MODEL_PATH
    sink_cache_dir = os.environ.get("OLMO3_SINK_CACHE_DIR", str(Path(RUN_DIR) / "olmo3_sink_cache"))
    active_path = prepare_olmo3_sink_model_path(
        MODEL_PATH,
        cache_dir=sink_cache_dir,
        sink_init_value=env_float("OLMO3_SINK_INIT_VALUE", -10.0),
        dtype=os.environ.get("OLMO3_SINK_DTYPE", "bfloat16"),
        sinks_npz=os.environ.get("OLMO3_SINKS_NPZ") or None,
        force_convert=env_bool("OLMO3_SINK_FORCE_CONVERT", False),
    )
    logger.info("Resolved OLMo3 sink model path: %s -> %s", MODEL_PATH, active_path)
    return active_path


ACTIVE_MODEL_PATH = resolve_model_path()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"Expected list JSON dataset: {path}")
    if suffix == ".parquet":
        try:
            import polars as pl

            return pl.read_parquet(path).to_dicts()
        except ImportError:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported dataset suffix: {suffix}")


def row_problem(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt", "theorem", "statement"):
        value = row.get(key)
        if value:
            return str(value).strip()
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages.strip()
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts).strip()
    return ""


def make_dataset() -> list[dict[str, Any]]:
    max_rows = env_int("MAX_ROWS", 0)
    records = load_records(DATASET_PATH)
    if max_rows > 0:
        records = records[:max_rows]
    dataset = []
    skipped = 0
    for index, row in enumerate(records):
        problem = row_problem(row)
        if not problem:
            skipped += 1
            continue
        dataset.append(
            {
                "prompt": build_deepseek_proof_generation_prompt(problem),
                "problem": problem,
                "question": problem,
                "solution": str(row.get("solution") or ""),
                "source_index": index,
            }
        )
    if not dataset:
        raise ValueError(f"No usable rows in {DATASET_PATH}; skipped={skipped}")
    logger.info("Loaded proof dataset: rows=%d skipped=%d path=%s", len(dataset), skipped, DATASET_PATH)
    return dataset


def load_model():
    sink_enabled = env_bool("OLMO3_USE_SINK", False)
    default_attn = "olmo3_sink_fa3" if sink_enabled else "flash_attention_2"
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION")
    if not attn_implementation:
        attn_implementation = os.environ.get("OLMO3_SINK_ATTN_IMPLEMENTATION", default_attn) if sink_enabled else default_attn
    return load_olmo3_causal_lm(
        ACTIVE_MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        gradient_checkpointing=env_bool("GRADIENT_CHECKPOINTING", True),
        use_attention_sink=False,
        sink_cache_dir=os.environ.get("OLMO3_SINK_CACHE_DIR", str(Path(RUN_DIR) / "olmo3_sink_cache")),
        sink_init_value=env_float("OLMO3_SINK_INIT_VALUE", -10.0),
        sinks_npz=os.environ.get("OLMO3_SINKS_NPZ") or None,
        sink_force_convert=env_bool("OLMO3_SINK_FORCE_CONVERT", False),
        sink_dtype=os.environ.get("OLMO3_SINK_DTYPE", "bfloat16"),
        apply_liger=env_bool("OLMO3_SINK_APPLY_LIGER", False),
    )


def main() -> None:
    training_config = TrainingConfig(
        model_id=ACTIVE_MODEL_PATH,
        training_mode="full",
        trainable_modules=None,
        learning_rate=env_float("LEARNING_RATE", 5e-7),
        weight_decay=env_float("WEIGHT_DECAY", 0.0),
        adam_beta1=env_float("ADAM_BETA1", 0.9),
        adam_beta2=env_float("ADAM_BETA2", 0.95),
        max_grad_norm=env_float("MAX_GRAD_NORM", 1.0),
        lr_scheduler_type=os.environ.get("LR_SCHEDULER_TYPE", "constant_with_warmup"),
        lr_warmup_steps=env_int("LR_WARMUP_STEPS", 10),
        ddis_eps_low=env_float("DDIS_EPS_LOW", 0.2),
        ddis_eps_high=env_float("DDIS_EPS_HIGH", 0.28),
        loss_backend=os.environ.get("LOSS_BACKEND", "torch"),
        loss_aggregation_mode=os.environ.get("LOSS_AGGREGATION_MODE", "seq-mean-token-mean"),
        num_inner_steps=env_int("NUM_INNER_STEPS", 1),
        max_capacity=env_int("MAX_CAPACITY", 70_000),
        pad_to_multiple_of_training=env_int("PAD_TO_MULTIPLE_OF_TRAINING", 1024),
        dataloader_num_workers=env_int("DATALOADER_NUM_WORKERS", 0),
        use_routing_replay=False,
        save_routing_indices=False,
        weight_sync_mode="merged",
        flush_cache_on_sync=env_bool("FLUSH_CACHE_ON_SYNC", False),
        flush_cache_every_n_steps=env_int("FLUSH_CACHE_EVERY_N_STEPS", 0),
        weight_sync_interval=env_int("WEIGHT_SYNC_INTERVAL", 10),
        min_weight_sync_secs=env_float("MIN_WEIGHT_SYNC_SECS", 60.0),
        max_staleness=env_int("MAX_STALENESS", 10),
        run_dir=RUN_DIR,
        save_checkpoint_every=env_int("SAVE_CHECKPOINT_EVERY", 10),
        save_latest_checkpoint=env_bool("SAVE_LATEST_CHECKPOINT", True),
        max_checkpoints_to_keep=env_int("MAX_CHECKPOINTS_TO_KEEP", 3),
        use_wandb=env_bool("USE_WANDB", True),
        wandb_project=os.environ.get("WANDB_PROJECT", "olmo3-32b-agent-factory"),
        wandb_run_name=os.environ.get("WANDB_RUN_NAME", Path(RUN_DIR).name),
        wandb_notes=os.environ.get("WANDB_NOTES", "OLMo3 proof RL with local verifier/meta reward."),
    )

    accelerator = Accelerator()
    logger.info("Accelerator: %s, processes=%d", accelerator.state.distributed_type, accelerator.num_processes)
    logger.info("Loading OLMo3 model: %s", ACTIVE_MODEL_PATH)
    model = load_model()
    model_trainer = ModelTrainer(config=training_config, accelerator=accelerator, model=model)

    reward = OLMoProofReward(
        OLMoProofRewardConfig(
            tokenizer_name_or_path=ACTIVE_MODEL_PATH,
            judge_urls=JUDGE_URLS,
            verify_n=env_int("VERIFY_N", 1),
            meta_n=env_int("META_N", 0),
            max_context_tokens=env_int("JUDGE_MAX_CONTEXT_TOKENS", 40_000),
            max_new_tokens=env_int("JUDGE_MAX_NEW_TOKENS", 40_000),
            context_margin_tokens=env_int("JUDGE_CONTEXT_MARGIN_TOKENS", 256),
            min_completion_tokens=env_int("JUDGE_MIN_COMPLETION_TOKENS", 128),
            temperature=env_float("JUDGE_TEMPERATURE", 0.7),
            top_p=env_float("JUDGE_TOP_P", 0.95),
            timeout=env_float("JUDGE_TIMEOUT", 1800.0),
            format_partial_score=env_float("FORMAT_PARTIAL_SCORE", 0.7),
            require_format=env_bool("REQUIRE_FORMAT", True),
            proof_weight=env_float("PROOF_WEIGHT", 0.76),
            self_eval_weight=env_float("SELF_EVAL_WEIGHT", 0.24),
            max_forwarded_self_evaluation_chars=env_int("MAX_FORWARDED_SELF_EVALUATION_CHARS", 12_000),
            preview_dir=os.environ.get("PROOF_REWARD_PREVIEW_DIR", str(Path(RUN_DIR) / "prompt_previews")),
            preview_max_chars=env_int("PROOF_REWARD_PREVIEW_MAX_CHARS", 240_000),
        )
    )

    flow = RLFlow(
        training_config=training_config,
        model_trainer=model_trainer,
        dataset=make_dataset(),
        reward_fn=per_rollout(reward),
        prompt_key="prompt",
        server_urls=SGLANG_URLS,
        num_workers=env_int("NUM_WORKERS", 8),
        worker_concurrency=env_int("WORKER_CONCURRENCY", 4),
        batch_size=env_int("BATCH_SIZE", 32),
        group_size=env_int("GROUP_SIZE", 8),
        loop_config=LoopConfig(
            backend="sglang",
            prompt_format="chat_template",
            tokenizer_name_or_path=ACTIVE_MODEL_PATH,
            sampling=SamplingParams(
                temperature=env_float("ROLLOUT_TEMPERATURE", 0.7),
                top_p=env_float("ROLLOUT_TOP_P", 0.95),
                top_k=env_int("ROLLOUT_TOP_K", -1),
                min_p=env_float("ROLLOUT_MIN_P", 0.0) or None,
            ),
            max_rounds=1,
            max_total_tokens=env_int("ROLLOUT_MAX_RESPONSE_TOKENS", 60_000),
            max_round_tokens=env_int("ROLLOUT_MAX_RESPONSE_TOKENS", 60_000),
            max_context_tokens=env_int("ROLLOUT_MAX_CONTEXT_TOKENS", 65_536),
            use_streaming=env_bool("ROLLOUT_STREAMING", False),
            max_aborts=env_int("MAX_ABORTS", 20),
        ),
        conv_config=ConversationConfig(),
        record_config=RecordConfig(logprobs=True, routing_indices=False, usage=True, entropy=env_bool("RECORD_ENTROPY", False)),
        normalize_advantages=env_bool("NORMALIZE_ADVANTAGES", False),
        filter_all_failed=env_bool("FILTER_ALL_FAILED", True),
        filter_all_solved=env_bool("FILTER_ALL_SOLVED", True),
        max_batches=env_int("MAX_BATCHES", 100),
        base_seed=env_int("BASE_SEED", 42),
        cache_salt_mode=os.environ.get("CACHE_SALT_MODE", "per_rollout"),
        server_affinity=env_bool("SERVER_AFFINITY", False),
    )

    def on_step(metrics: dict[str, float], step: int) -> None:
        logger.info(
            "Step %d: loss=%.4f reward_mean=%.4f verifier=%.4f meta=%.4f tokens/s=%.1f",
            step,
            metrics.get("loss", float("nan")),
            metrics.get("reward/mean", float("nan")),
            metrics.get("reward_components/verifier_mean", float("nan")),
            metrics.get("reward_components/meta_mean", float("nan")),
            metrics.get("train/tokens_per_sec", float("nan")),
        )

    flow.run(on_step=on_step)


if __name__ == "__main__":
    main()
