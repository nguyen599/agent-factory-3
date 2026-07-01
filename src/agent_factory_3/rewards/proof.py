"""OLMo3 proof reward utilities.

The prompts and parsing contract mirror the proof pipeline used in
`submissions-instructions/src/run.py`: a candidate should produce
`## Solution`, optionally `## Self Evaluation`, and boxed verifier scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from ..rollout.parallel.config import RolloutResult

logger = logging.getLogger(__name__)

EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""

_SCORE_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}|"
    r"\bboxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}",
    flags=re.IGNORECASE,
)
_FALLBACK_SCORE_PATTERN = re.compile(
    r"(?:final\s+overall\s+score|score|rating)\s*(?:should\s+be|is|:)?\s*"
    r"\**\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\**\s*\.?\s*$",
    flags=re.IGNORECASE,
)
_THINK_BLOCK_PATTERN = re.compile(r"(?is)<think>.*?</think>\s*")
_HEADER_SUFFIX_PATTERN = r"[ \t]*:?[ \t]*$"
_VISIBLE_OUTPUT_MARKERS = (
    "# Solution",
    "## Solution",
    "# Self Evaluation",
    "## Self Evaluation",
    "# Self Evaluate",
    "## Self Evaluate",
    "Based on my evaluation",
    "Based on my analysis",
    "Here is my evaluation",
    "Here is my analysis",
)
_FATAL_FORMAT_ERRORS = frozenset({"missing_solution_heading", "empty_solution"})


@dataclass
class OLMoProofRewardConfig:
    tokenizer_name_or_path: str
    judge_urls: list[str]
    verify_n: int = 1
    meta_n: int = 0
    max_context_tokens: int = 40_000
    max_new_tokens: int = 40_000
    context_margin_tokens: int = 256
    min_completion_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: float = 1800.0
    format_partial_score: float = 0.7
    require_format: bool = True
    proof_weight: float = 0.76
    self_eval_weight: float = 0.24
    max_forwarded_self_evaluation_chars: int = 12_000
    preview_dir: str | None = None
    preview_max_chars: int = 240_000


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_deepseek_proof_generation_prompt(question: str) -> str:
    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
```

In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.
- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation // Your evaluation of your own solution above should start with this exact same markdown title

Here is my evaluation of the solution: // Your analysis should start with this exact same phrase
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above

---

Here is your task input:

## Problem
{question}"""


def build_deepseek_proof_verification_prompt(question: str, proof: str) -> str:
    return f"""## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{question}

## Solution
{proof}"""


def build_deepseek_meta_verification_prompt(question: str, proof: str, proof_analysis: str) -> str:
    return f"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
{EVALUATION_RUBRIC}
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.
2. The most important part is defect analysis: check whether errors or defects pointed out by the "solution evaluation" are reasonable.
3. If at least one defect found by the "solution evaluation" is unreasonable, rate it 0 or 0.5 depending on whether all or only some defects are unreasonable.
4. If no defects are found, or all found defects are reasonable, rate expression and score analysis. If these are reasonable, rate it 1; otherwise 0.5.

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""


def strip_reasoning_blocks(text: str) -> str:
    raw = str(text or "")
    cleaned = _THINK_BLOCK_PATTERN.sub("", raw)
    orphan_close = re.search(r"(?is)</think>\s*", cleaned)
    if orphan_close is not None:
        first_marker = min(
            (
                idx
                for marker in _VISIBLE_OUTPUT_MARKERS
                if (idx := cleaned.lower().find(marker.lower())) >= 0
            ),
            default=None,
        )
        if first_marker is None or orphan_close.start() < first_marker:
            prefix = cleaned[: orphan_close.start()]
            suffix = cleaned[orphan_close.end() :]
            if "<think" not in prefix.lower():
                cleaned = suffix
    return cleaned.strip()


def extract_boxed_score(text: str) -> Optional[float]:
    score_text = str(text or "")[-2000:]
    matches = list(_SCORE_PATTERN.finditer(score_text))
    if not matches:
        matches = list(_FALLBACK_SCORE_PATTERN.finditer(score_text.strip()))
    if not matches:
        return None
    try:
        groups = [group for group in matches[-1].groups() if group is not None]
        return float(groups[0])
    except (IndexError, TypeError, ValueError):
        return None


def _header_matches(text: str, header: str) -> list[re.Match[str]]:
    header_name = re.sub(r"^#+[ \t]*", "", header.strip()).strip()
    header_names = [header_name]
    if header_name.lower().replace("-", " ") == "self evaluation":
        header_names.extend(["Self Evaluate", "Self-Evaluate", "Self-Evaluation"])
    matches: list[re.Match[str]] = []
    for name in header_names:
        name_pattern = re.escape(name).replace(r"\ ", r"[ \t]+").replace(r"\-", r"[- ]")
        matches.extend(
            re.finditer(
                rf"(?im)^[ \t]*(?:#{{1,6}}[ \t]*)?{name_pattern}{_HEADER_SUFFIX_PATTERN}",
                text,
            )
        )
    unique = {(match.start(), match.end()): match for match in matches}
    return [unique[key] for key in sorted(unique)]


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def parse_generation_response(text: str, *, require_self_evaluation: bool = True) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    raw_chars = len(text or "")
    solution_headers = _header_matches(visible, "## Solution")
    evaluation_headers = _header_matches(visible, "## Self Evaluation")
    format_errors: list[str] = []
    if not solution_headers:
        format_errors.append("missing_solution_heading")
        proof = visible.strip()
        if not proof:
            format_errors.append("empty_solution")
        return {
            "proof": proof,
            "self_evaluation": "",
            "self_score": None,
            "has_solution_section": False,
            "has_self_evaluation_section": False,
            "is_valid_candidate_response": False,
            "format_ok": False,
            "format_errors": format_errors,
            "raw_chars": raw_chars,
            "visible_chars": len(visible),
        }
    solution_header = solution_headers[-1]
    following_evaluation = next(
        (match for match in evaluation_headers if match.start() > solution_header.end()),
        None,
    )
    if following_evaluation is None:
        proof = visible[solution_header.end() :].strip()
        self_evaluation = ""
        has_self_evaluation = False
    else:
        proof = visible[solution_header.end() : following_evaluation.start()].strip()
        self_evaluation = visible[following_evaluation.end() :].strip()
        has_self_evaluation = True
    self_score = extract_boxed_score(self_evaluation)
    if not proof:
        format_errors.append("empty_solution")
    if not has_self_evaluation:
        format_errors.append("missing_self_evaluation_heading")
    if self_evaluation and not _contains_any_phrase(
        self_evaluation,
        (
            "Here is my evaluation of the solution:",
            "Detailed evaluation:",
            "Detailed evaluation",
        ),
    ):
        format_errors.append("missing_evaluation_phrase")
    if self_evaluation and not _contains_any_phrase(
        self_evaluation,
        (
            "Based on my evaluation, the final overall score should be:",
            "Based on my evaluation, the final overall score is:",
            "final overall score should be:",
            "final overall score is:",
        ),
    ):
        format_errors.append("missing_score_phrase")
    if self_score is None:
        format_errors.append("missing_or_invalid_boxed_self_score")
    valid = bool(proof and (has_self_evaluation or not require_self_evaluation))
    return {
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": True,
        "has_self_evaluation_section": has_self_evaluation,
        "is_valid_candidate_response": valid,
        "format_ok": not format_errors,
        "format_errors": format_errors,
        "raw_chars": raw_chars,
        "visible_chars": len(visible),
    }


def _clip_middle_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 256:
        return text[:max_chars]
    keep_left = max_chars // 2
    keep_right = max_chars - keep_left
    return text[:keep_left] + "\n...[middle clipped for judge context]...\n" + text[-keep_right:]


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


class OLMoProofReward:
    """Per-rollout proof reward backed by local SGLang judge calls."""

    def __init__(self, config: OLMoProofRewardConfig):
        if not config.judge_urls:
            raise ValueError("OLMoProofRewardConfig.judge_urls must not be empty")
        proof_weight = max(0.0, float(config.proof_weight))
        self_eval_weight = max(0.0, float(config.self_eval_weight))
        total_weight = proof_weight + self_eval_weight
        if total_weight <= 0:
            raise ValueError("At least one of proof_weight or self_eval_weight must be positive")
        if abs(total_weight - 1.0) > 1e-6:
            config.proof_weight = proof_weight / total_weight
            config.self_eval_weight = self_eval_weight / total_weight
        self.config = config
        self.tokenizer = _load_tokenizer(config.tokenizer_name_or_path)
        self._counter = count()
        self._counter_lock = threading.Lock()

    def __call__(self, result: "RolloutResult", metadata: dict[str, Any]) -> tuple[float, dict[str, float]]:
        if not result.success or result.result is None:
            return 0.0, {"reward_valid": 0.0, "format_score": 0.0}
        question = str(metadata.get("problem") or metadata.get("question") or metadata.get("prompt") or "")
        candidate_text = self._extract_assistant_text(result)
        parsed = parse_generation_response(candidate_text, require_self_evaluation=True)
        rollout_name = self._rollout_name(metadata)
        self._write_preview(
            "rollout_response",
            rollout_name,
            candidate_text,
            {
                "raw_chars": parsed.get("raw_chars"),
                "visible_chars": parsed.get("visible_chars"),
                "format_errors": parsed.get("format_errors", []),
            },
        )
        proof = str(parsed["proof"])
        format_errors = [str(error) for error in parsed.get("format_errors", [])]
        fatal_format_errors = [error for error in format_errors if error in _FATAL_FORMAT_ERRORS]
        if not proof or (self.config.require_format and fatal_format_errors):
            return 0.0, {
                "reward_valid": 1.0,
                "format_score": 0.0,
                "proof_chars": float(len(proof)),
                "verifier_mean": 0.0,
                "meta_mean": 0.0,
                "format_ok": 0.0,
                "fatal_format": 1.0,
            }

        format_ok = bool(parsed.get("format_ok"))
        format_score = 1.0 if format_ok else max(0.0, min(1.0, float(self.config.format_partial_score)))

        verifier_scores: list[float] = []
        meta_scores: list[float] = []
        for verify_idx in range(max(1, self.config.verify_n)):
            verify_prompt = build_deepseek_proof_verification_prompt(question, proof)
            self._write_preview("proof_judge_prompt", f"{rollout_name}_v{verify_idx}", verify_prompt)
            verify_text = self._generate_text(verify_prompt, stage=f"verify_{verify_idx}")
            self._write_preview("proof_judge_response", f"{rollout_name}_v{verify_idx}", verify_text)
            verifier_score = extract_boxed_score(verify_text) or 0.0
            verifier_score = max(0.0, min(1.0, verifier_score))
            verifier_scores.append(verifier_score)

        self_evaluation = str(parsed["self_evaluation"] or "")
        forwarded_self_evaluation = _clip_middle_text(
            self_evaluation,
            max(0, int(self.config.max_forwarded_self_evaluation_chars)),
        )
        if self.config.meta_n > 0 and forwarded_self_evaluation:
            for meta_idx in range(self.config.meta_n):
                meta_prompt = build_deepseek_meta_verification_prompt(question, proof, forwarded_self_evaluation)
                self._write_preview("meta_judge_prompt", f"{rollout_name}_m{meta_idx}", meta_prompt)
                meta_text = self._generate_text(meta_prompt, stage=f"meta_{meta_idx}")
                self._write_preview("meta_judge_response", f"{rollout_name}_m{meta_idx}", meta_text)
                meta_score = extract_boxed_score(meta_text) or 0.0
                meta_score = max(0.0, min(1.0, meta_score))
                meta_scores.append(meta_score)

        proof_score = _mean(verifier_scores)
        meta_mean = _mean(meta_scores, default=1.0 if self.config.meta_n <= 0 else 0.0)
        self_score = parsed["self_score"]
        if self.config.meta_n <= 0:
            # Without a meta-verifier we cannot reliably reward self-evaluation
            # quality, so use proof-score-only reward and keep self-score as a metric.
            score_alignment = 0.0 if self_score is None else max(0.0, 1.0 - abs(float(self_score) - proof_score))
            self_eval_score = 0.0
            base_reward = proof_score
        else:
            score_alignment = 0.0 if self_score is None else max(0.0, 1.0 - abs(float(self_score) - proof_score))
            self_eval_score = meta_mean
            base_reward = (
                self.config.proof_weight * proof_score
                + self.config.self_eval_weight * score_alignment * self_eval_score
            )
        reward = max(0.0, min(1.0, base_reward * format_score))
        components = {
            "reward_valid": 1.0,
            "deepseekmath_v2_reward": float(reward),
            "deepseekmath_v2_correct_rate": 1.0 if reward >= 0.999 else 0.0,
            "verifiable_reward": float(proof_score),
            "verifiable_correct_rate": 1.0 if proof_score >= 0.999 else 0.0,
            "format_score": float(format_score),
            "format_ok": 1.0 if format_ok else 0.0,
            "verifier_mean": float(proof_score),
            "meta_mean": float(meta_mean),
            "proof_score": float(proof_score),
            "judge_mean": float(proof_score),
            "self_eval_score": float(self_eval_score),
            "score_alignment": float(score_alignment),
            "base_reward": float(base_reward),
            "proof_chars": float(len(proof)),
            "self_eval_chars": float(len(str(parsed["self_evaluation"]))),
            "self_score": -1.0 if parsed["self_score"] is None else float(parsed["self_score"]),
            "meta_enabled": 1.0 if self.config.meta_n > 0 else 0.0,
            "solved": 1.0 if reward >= 0.999 else 0.0,
        }
        logger.info(
            "DeepSeekMath-V2 proof reward rollout=%s reward=%.4f format_score=%.3f "
            "proof_score=%.3f self_score=%s meta=%.3f alignment=%.3f errors=%s",
            rollout_name,
            reward,
            format_score,
            proof_score,
            parsed["self_score"],
            meta_mean,
            score_alignment,
            format_errors,
        )
        return reward, components

    def _extract_assistant_text(self, result: "RolloutResult") -> str:
        react = result.result
        assert react is not None
        chunks = []
        for step in react.get_assistant_steps():
            usage = step.usage or {}
            text = usage.get("assistant_text")
            if isinstance(text, str):
                chunks.append(text)
            else:
                chunks.append(self.tokenizer.decode(react.tokens[step.start : step.end], skip_special_tokens=True))
        return "\n".join(chunks).strip()

    def _next_url(self) -> str:
        with self._counter_lock:
            index = next(self._counter)
        return self.config.judge_urls[index % len(self.config.judge_urls)].rstrip("/")

    def _rollout_name(self, metadata: dict[str, Any]) -> str:
        source = "|".join(
            str(metadata.get(key, ""))
            for key in ("group_id", "source_index", "problem", "question", "prompt")
        )
        digest = hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"rollout_{digest}"

    def _write_preview(
        self,
        stage: str,
        name: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.preview_dir:
            return
        try:
            preview_dir = Path(self.config.preview_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "sample"))[:120]
            digest = hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
            path = preview_dir / f"{stage}_{safe_name}_{digest}.txt"
            clipped = str(text or "")
            was_clipped = False
            if self.config.preview_max_chars > 0 and len(clipped) > self.config.preview_max_chars:
                clipped = _clip_middle_text(clipped, self.config.preview_max_chars)
                was_clipped = True
            payload = {
                "stage": stage,
                "name": name,
                "chars": len(str(text or "")),
                "clipped": was_clipped,
            }
            if metadata:
                payload.update(metadata)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n\n" + clipped,
                encoding="utf-8",
            )
            logger.info("Prompt preview written stage=%s name=%s path=%s chars=%d clipped=%s", stage, name, path, len(str(text or "")), was_clipped)
        except Exception:
            logger.exception("Failed to write prompt preview stage=%s name=%s", stage, name)

    def _render_prompt(self, prompt: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]

    def _generate_text(self, prompt: str, *, stage: str) -> str:
        budget_limit = self.config.max_context_tokens - self.config.context_margin_tokens - self.config.min_completion_tokens
        clipped_prompt = prompt
        prompt_tokens = self._render_prompt(clipped_prompt)
        for _ in range(4):
            if len(prompt_tokens) <= budget_limit:
                break
            ratio = max(0.1, budget_limit / max(len(prompt_tokens), 1))
            next_chars = max(2048, int(len(clipped_prompt) * ratio * 0.9))
            clipped_prompt = _clip_middle_text(clipped_prompt, next_chars)
            prompt_tokens = self._render_prompt(clipped_prompt)
        while len(prompt_tokens) >= self.config.max_context_tokens - self.config.context_margin_tokens and len(clipped_prompt) > 2048:
            clipped_prompt = _clip_middle_text(clipped_prompt, max(2048, len(clipped_prompt) // 2))
            prompt_tokens = self._render_prompt(clipped_prompt)
        available_completion = self.config.max_context_tokens - self.config.context_margin_tokens - len(prompt_tokens)
        max_new_tokens = max(1, min(self.config.max_new_tokens, available_completion))
        stop_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(int(self.tokenizer.eos_token_id))
        payload = {
            "input_ids": prompt_tokens,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": -1,
                "stop_token_ids": sorted(set(stop_ids)),
            },
            "stream": False,
            "return_logprob": False,
        }
        url = f"{self._next_url()}/generate"
        started = time.time()
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        output_ids = [int(token_id) for token_id in data.get("output_ids", [])]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        logger.info(
            "Proof reward judge stage=%s prompt_tokens=%d max_new_tokens=%d output_tokens=%d elapsed=%.1fs",
            stage,
            len(prompt_tokens),
            max_new_tokens,
            len(output_ids),
            time.time() - started,
        )
        return text
