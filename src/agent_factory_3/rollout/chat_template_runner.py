"""Single-turn HF chat-template rollout runner.

This runner is intentionally simpler than the Harmony ReAct runner: it renders
one system/user chat prompt with a Hugging Face tokenizer, generates one
assistant completion, and records token/logprob arrays compatible with the
existing DDIS trainer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import Optional

from openai_harmony import Conversation, Message, Role
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import LoopConfig, RecordConfig
from .llm_backend import LLMBackend
from .types import AssistantStep, EndReason, InitStep, ReactResult, TokenBudget, TruncationReason, WeightSegment

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class ChatTemplateRunner:
    """Run a one-shot chat-template generation for non-Harmony models."""

    def __init__(self, config: LoopConfig, llm_backend: LLMBackend):
        if not config.tokenizer_name_or_path:
            raise ValueError("LoopConfig.tokenizer_name_or_path is required for prompt_format='chat_template'")
        self.config = config
        self.llm_backend = llm_backend
        self.tokenizer = _load_tokenizer(config.tokenizer_name_or_path)

    async def run(
        self,
        user_prompt: str,
        *,
        stop_event: Optional[asyncio.Event] = None,
        record_config: Optional[RecordConfig] = None,
        rollout_id: Optional[str] = None,
    ) -> ReactResult:
        rid = f"[{(rollout_id or 'default'):14s}]"
        messages = []
        if self.config.chat_template_system_prompt:
            messages.append({"role": "system", "content": self.config.chat_template_system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.config.chat_template_add_generation_prompt,
        )
        prompt_tokens = [int(token_id) for token_id in prompt_tokens]
        budget = self._calculate_token_budget(len(prompt_tokens))
        if not budget.can_generate:
            raise ValueError(
                f"Prompt exceeds chat-template context budget: prompt_tokens={len(prompt_tokens)} "
                f"max_context_tokens={self.config.max_context_tokens}"
            )

        logger.info("%s START  | prompt_tokens=%d max_tokens=%d", rid, len(prompt_tokens), budget.max_tokens)
        start_time = time.time()
        generation = await self.llm_backend.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=budget.max_tokens,
            stop_token_ids=self._stop_token_ids(),
            sampling=self.config.sampling,
            stream=self.config.use_streaming,
            stop_event=stop_event,
            record=record_config,
        )

        completion_tokens = [int(token_id) for token_id in generation.token_ids]
        assistant_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        all_tokens = prompt_tokens + completion_tokens

        logprobs = None
        if record_config is not None and record_config.logprobs:
            completion_logprobs = generation.logprobs or [None] * len(completion_tokens)
            logprobs = [None] * len(prompt_tokens) + completion_logprobs

        top_logprobs = None
        if record_config is not None and record_config.top_logprobs > 0:
            completion_top_logprobs = generation.top_logprobs or [None] * len(completion_tokens)
            top_logprobs = [None] * len(prompt_tokens) + completion_top_logprobs

        entropy = None
        if record_config is not None and record_config.entropy:
            completion_entropy = generation.entropy or [None] * len(completion_tokens)
            entropy = [None] * len(prompt_tokens) + completion_entropy

        now = time.time()
        init_step = InitStep(
            start=0,
            end=len(prompt_tokens),
            message_start=0,
            message_end=len(messages),
            created_at=start_time,
            num_user_tokens=len(prompt_tokens),
        )
        assistant_step = AssistantStep(
            start=len(prompt_tokens),
            end=len(all_tokens),
            message_start=len(messages),
            message_end=len(messages) + 1,
            round_index=0,
            created_at=start_time,
            elapsed=now - start_time,
            stop_reason=generation.finish_reason,
            truncation_reason=budget.limiting_factor if generation.finish_reason == "length" else None,
            usage={
                "assistant_text": assistant_text,
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "backend_usage": generation.usage,
            },
            weight_segments=[
                WeightSegment(
                    start=0,
                    end=len(completion_tokens),
                    weight_version=generation.weight_version or "0",
                )
            ],
        )

        conversation_messages = [
            Message.from_role_and_content(Role.USER, user_prompt),
            Message.from_role_and_content(Role.ASSISTANT, assistant_text),
        ]
        if self.config.chat_template_system_prompt:
            conversation_messages.insert(
                0,
                Message.from_role_and_content(Role.SYSTEM, self.config.chat_template_system_prompt),
            )

        end_reason = EndReason.TOKEN_LIMIT if generation.finish_reason == "length" else EndReason.COMPLETED
        end_reason_detail = "token_limit:generation" if end_reason == EndReason.TOKEN_LIMIT else "completed:chat_template"
        result = ReactResult(
            end_reason=end_reason,
            end_reason_detail=end_reason_detail,
            errors=[],
            tokens=all_tokens,
            logprobs=logprobs,
            entropy=entropy,
            top_logprobs=top_logprobs,
            routing_indices=None,
            steps=[init_step, assistant_step],
            conversation=Conversation.from_messages(conversation_messages),
            total_tool_time=0.0,
            num_generated_tokens=len(completion_tokens),
            abort_records=[],
        )
        logger.info(
            "%s END    | reason=%s completion_tokens=%d elapsed=%.1fs",
            rid,
            result.end_reason.value,
            len(completion_tokens),
            result.steps[-1].elapsed or 0.0,
        )
        return result

    def _calculate_token_budget(self, prompt_len: int) -> TokenBudget:
        context_space = self.config.max_context_tokens - prompt_len
        max_tokens = min(self.config.max_round_tokens, self.config.max_total_tokens, context_space)
        if max_tokens == context_space:
            reason = TruncationReason.CONTEXT_SPACE
        elif max_tokens == self.config.max_total_tokens:
            reason = TruncationReason.GENERATION_QUOTA
        else:
            reason = TruncationReason.ROUND_LIMIT
        return TokenBudget(max_tokens=max(0, max_tokens), limiting_factor=reason)

    def _stop_token_ids(self) -> list[int]:
        configured = list(self.config.stop_token_ids or [])
        candidates = configured
        if self.tokenizer.eos_token_id is not None:
            candidates.append(int(self.tokenizer.eos_token_id))
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                candidates.append(token_id)
        return sorted(set(candidates))
