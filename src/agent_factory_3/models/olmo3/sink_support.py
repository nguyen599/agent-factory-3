"""OLMo3 attention-sink helpers for training and SGLang rollout serving."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _candidate_olmo3_sink_parents() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if os.environ.get("OLMO3_SINK_PATH"):
        path = Path(os.environ["OLMO3_SINK_PATH"]).expanduser().resolve()
        candidates.append(path.parent if path.name == "olmo3_sink" else path)
    for parent in here.parents:
        candidates.extend(
            [
                parent / "olmo3_sink",
                parent.parent / "olmo3_sink",
                parent / "submissions-instructions" / "src",
                parent.parent / "submissions-instructions" / "src",
            ]
        )
    candidates.extend([Path.cwd() / "olmo3_sink", Path.cwd() / "submissions-instructions" / "src"])

    parents: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        package = candidate if candidate.name == "olmo3_sink" else candidate / "olmo3_sink"
        if not (package / "__init__.py").exists():
            continue
        package_parent = package.parent.resolve()
        key = str(package_parent)
        if key not in seen:
            parents.append(package_parent)
            seen.add(key)
    return parents


def ensure_olmo3_sink_importable() -> None:
    """Make the local `olmo3_sink` package importable and register Auto classes."""
    for parent in _candidate_olmo3_sink_parents():
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
    try:
        from olmo3_sink import register_olmo3_sink
    except Exception as exc:
        searched = ", ".join(str(p) for p in _candidate_olmo3_sink_parents()) or "<none>"
        raise RuntimeError(
            "Could not import olmo3_sink. Set OLMO3_SINK_PATH to the repo/package path. "
            f"Searched parents: {searched}"
        ) from exc
    register_olmo3_sink()


def _read_config(model_path: str | Path) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def is_olmo3_sink_checkpoint(model_path: str | Path) -> bool:
    return _read_config(model_path).get("model_type") == "olmo3_sink"


def _has_weight_files(model_path: str | Path) -> bool:
    path = Path(model_path)
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    return any(path.glob(pattern) for pattern in patterns)


def _stable_name(model_path: str, sink_init_value: float, sinks_npz: str | None) -> str:
    digest_input = json.dumps(
        {
            "model_path": model_path,
            "sink_init_value": sink_init_value,
            "sinks_npz": sinks_npz or "",
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    stem = Path(model_path.rstrip("/")).name or "olmo3"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in stem)[:80]
    return f"{safe}-olmo3-sink-{digest}"


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                pass


def _completed_sink_checkpoint(path: Path) -> bool:
    return (path / ".olmo3_sink_complete").exists() and is_olmo3_sink_checkpoint(path) and _has_weight_files(path)


def prepare_olmo3_sink_model_path(
    model_path: str,
    *,
    cache_dir: str | Path,
    sink_init_value: float = -10.0,
    dtype: str = "bfloat16",
    sinks_npz: str | None = None,
    force_convert: bool = False,
) -> str:
    """Return an `olmo3_sink` HF checkpoint path, converting stock OLMo3 if needed."""
    ensure_olmo3_sink_importable()
    source = str(model_path)
    if is_olmo3_sink_checkpoint(source) and not force_convert:
        logger.info("Using existing olmo3_sink checkpoint: %s", source)
        return source

    cache_root = Path(cache_dir).expanduser().resolve()
    dst = cache_root / _stable_name(source, sink_init_value, sinks_npz)
    marker = dst / ".olmo3_sink_complete"
    lock_path = cache_root / f"{dst.name}.lock"

    if _completed_sink_checkpoint(dst) and not force_convert:
        logger.info("Using cached olmo3_sink checkpoint: %s", dst)
        return str(dst)

    with _file_lock(lock_path):
        if _completed_sink_checkpoint(dst) and not force_convert:
            return str(dst)
        if force_convert and dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        logger.info("Converting OLMo3 checkpoint to olmo3_sink: src=%s dst=%s", source, dst)
        from olmo3_sink.convert import convert

        convert(source, str(dst), sink_init_value=sink_init_value, dtype=dtype, sinks_npz=sinks_npz)
        marker.write_text("ok\n", encoding="utf-8")
    return str(dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.suffix == ".safetensors":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _rewrite_sglang_config(config: dict) -> dict:
    cfg = dict(config)
    rope_parameters = cfg.pop("rope_parameters", None)
    cfg.pop("auto_map", None)
    cfg["model_type"] = "olmo3"
    cfg["architectures"] = ["Olmo3SinkForCausalLM"]
    cfg["dtype"] = "bfloat16"
    cfg["torch_dtype"] = "bfloat16"
    cfg["use_cache"] = True
    if rope_parameters:
        cfg["rope_theta"] = rope_parameters.get("rope_theta", cfg.get("rope_theta"))
        cfg["rope_scaling"] = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        pattern = [1 if layer_type == "sliding_attention" else 0 for layer_type in layer_types]
        cfg["is_hybrid_swa"] = True
        cfg["hybrid_layer_pattern"] = pattern
    return cfg


def _completed_sglang_deploy(path: Path) -> bool:
    cfg = _read_config(path)
    return (
        (path / ".olmo3_sink_sglang_complete").exists()
        and cfg.get("model_type") == "olmo3"
        and cfg.get("architectures") == ["Olmo3SinkForCausalLM"]
        and _has_weight_files(path)
    )


def prepare_sglang_olmo3_sink_deploy_dir(
    model_path: str,
    *,
    cache_dir: str | Path,
    deploy_dir: str | Path | None = None,
    sink_init_value: float = -10.0,
    dtype: str = "bfloat16",
    sinks_npz: str | None = None,
    force_convert: bool = False,
    force_deploy: bool = False,
) -> str:
    """Build an SGLang-compatible deploy view for an `olmo3_sink` checkpoint."""
    sink_model = Path(
        prepare_olmo3_sink_model_path(
            model_path,
            cache_dir=cache_dir,
            sink_init_value=sink_init_value,
            dtype=dtype,
            sinks_npz=sinks_npz,
            force_convert=force_convert,
        )
    )
    deploy = Path(deploy_dir).expanduser().resolve() if deploy_dir else sink_model.with_name(f"{sink_model.name}-sglang")
    lock_path = deploy.with_suffix(deploy.suffix + ".lock")
    if _completed_sglang_deploy(deploy) and not force_deploy:
        logger.info("Using cached SGLang olmo3_sink deploy dir: %s", deploy)
        return str(deploy)

    with _file_lock(lock_path):
        if _completed_sglang_deploy(deploy) and not force_deploy:
            return str(deploy)
        if force_deploy and deploy.exists():
            shutil.rmtree(deploy)
        deploy.mkdir(parents=True, exist_ok=True)
        cfg = _rewrite_sglang_config(_read_config(sink_model))
        (deploy / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        for src in sink_model.iterdir():
            if src.name in {"config.json", ".olmo3_sink_complete"} or src.name.startswith("_resume"):
                continue
            if src.is_dir():
                continue
            _link_or_copy(src, deploy / src.name)
        (deploy / ".olmo3_sink_sglang_complete").write_text("ok\n", encoding="utf-8")
        logger.info("Prepared SGLang olmo3_sink deploy dir: %s", deploy)
    return str(deploy)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OLMo3 attention-sink checkpoints/deploy dirs.")
    sub = parser.add_subparsers(dest="command", required=True)

    model = sub.add_parser("prepare-model")
    model.add_argument("--model-path", required=True)
    model.add_argument("--cache-dir", required=True)
    model.add_argument("--sink-init-value", type=float, default=-10.0)
    model.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    model.add_argument("--sinks-npz")
    model.add_argument("--force-convert", action="store_true")

    deploy = sub.add_parser("prepare-sglang")
    deploy.add_argument("--model-path", required=True)
    deploy.add_argument("--cache-dir", required=True)
    deploy.add_argument("--deploy-dir")
    deploy.add_argument("--sink-init-value", type=float, default=-10.0)
    deploy.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    deploy.add_argument("--sinks-npz")
    deploy.add_argument("--force-convert", action="store_true")
    deploy.add_argument("--force-deploy", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.command == "prepare-model":
        path = prepare_olmo3_sink_model_path(
            args.model_path,
            cache_dir=args.cache_dir,
            sink_init_value=args.sink_init_value,
            dtype=args.dtype,
            sinks_npz=args.sinks_npz,
            force_convert=args.force_convert,
        )
    else:
        path = prepare_sglang_olmo3_sink_deploy_dir(
            args.model_path,
            cache_dir=args.cache_dir,
            deploy_dir=args.deploy_dir,
            sink_init_value=args.sink_init_value,
            dtype=args.dtype,
            sinks_npz=args.sinks_npz,
            force_convert=args.force_convert,
            force_deploy=args.force_deploy,
        )
    print(path)


if __name__ == "__main__":
    _main()
