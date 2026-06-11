"""Tests for vision_config orphan stripping in diffusion_gemma VLM loading.

Background: oQ-quantized diffusion_gemma checkpoints sometimes ship with
`vision_config` in `config.json` but no `model.encoder.vision_tower.*` or
`model.encoder.embed_vision.*` weights in the safetensors. Loading them via
`mlx_vlm.utils.load(...)` then crashes because mlx-vlm instantiates
`VisionEncoder` based on `vision_config`, and `load_weights(strict=True)`
fails with missing parameters. The `_strip_diffusion_gemma_vision_config_if_orphaned`
context manager swaps `mlx_vlm.utils.load_config` for the duration of the
call so that `vision_config` is set to None when vision weights are absent,
letting the model load as text-only.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import mlx_vlm.utils as _vu

from omlx.engine.vlm import (
    _has_diffusion_gemma_vision_weights,
    _strip_diffusion_gemma_vision_config_if_orphaned,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_safetensors(path: Path, keys: list[str]) -> None:
    from safetensors.numpy import save_file
    import numpy as np

    payload = {k: np.zeros((1,), dtype=np.float32) for k in keys}
    save_file(payload, str(path))


def _build_model_dir(
    tmp_path: Path,
    *,
    name: str,
    has_vision_config: bool,
    has_vision_weights: bool,
    model_type: str = "diffusion_gemma",
) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()

    config: dict = {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": model_type,
        "text_config": {"model_type": "diffusion_gemma_text", "hidden_size": 32},
    }
    if has_vision_config:
        config["vision_config"] = {"hidden_size": 16, "model_type": "gemma4_vision"}
        config["image_token_id"] = 258880
        config["boi_token_id"] = 255999
        config["eoi_token_id"] = 258882
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["model.decoder.layers.0.self_attn.q_proj.weight"]
    if has_vision_weights:
        keys.append("model.encoder.vision_tower.layers.0.attn.q_proj.weight")
        keys.append("model.encoder.embed_vision.embedding_projection.weight")
    _write_safetensors(model_dir / "model.safetensors", keys)

    return model_dir


# ---------------------------------------------------------------------------
# _has_diffusion_gemma_vision_weights
# ---------------------------------------------------------------------------


class TestHasDiffusionGemmaVisionWeights:
    def test_returns_true_when_vision_tower_key_present(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m1", has_vision_config=True, has_vision_weights=True,
        )
        assert _has_diffusion_gemma_vision_weights(model_dir) is True

    def test_returns_false_when_no_vision_keys(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m2", has_vision_config=True, has_vision_weights=False,
        )
        assert _has_diffusion_gemma_vision_weights(model_dir) is False

    def test_returns_false_for_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _has_diffusion_gemma_vision_weights(empty) is False

    def test_detects_embed_vision_prefix(self, tmp_path: Path):
        model_dir = tmp_path / "embed_only"
        model_dir.mkdir()
        config = {"model_type": "diffusion_gemma", "text_config": {}}
        (model_dir / "config.json").write_text(json.dumps(config))
        _write_safetensors(
            model_dir / "model.safetensors",
            ["model.encoder.embed_vision.proj.weight"],
        )
        assert _has_diffusion_gemma_vision_weights(model_dir) is True


# ---------------------------------------------------------------------------
# _strip_diffusion_gemma_vision_config_if_orphaned
# ---------------------------------------------------------------------------


class TestStripDiffusionGemmaVisionConfigIfOrphaned:
    def test_passthrough_for_non_diffusion_gemma_model(self, tmp_path: Path):
        # Non-diffusion_gemma models must be left untouched even when vision weights
        # are absent — the stripping is model-type-scoped.
        model_dir = _build_model_dir(
            tmp_path,
            name="gemma4",
            has_vision_config=True,
            has_vision_weights=False,
            model_type="gemma4",
        )
        with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert cfg.get("vision_config") is not None

    def test_passthrough_when_vision_config_already_none(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="no_vc", has_vision_config=False, has_vision_weights=False,
        )
        with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert cfg.get("vision_config") is None

    def test_passthrough_when_vision_weights_present(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="full", has_vision_config=True, has_vision_weights=True,
        )
        with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert cfg.get("vision_config") is not None

    def test_strips_vision_config_when_weights_missing(self, tmp_path: Path, caplog):
        # oQ-style checkpoint: vision_config present, vision weights absent.
        model_dir = _build_model_dir(
            tmp_path, name="orphaned", has_vision_config=True, has_vision_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
                cfg = _vu.load_config(model_dir)
        # vision_config must be explicitly None so mlx-vlm's
        # `setdefault("vision_config", {})` does not repopulate it.
        assert "vision_config" in cfg
        assert cfg["vision_config"] is None
        # WARN log fired.
        assert any(
            "vision_tower weights missing" in rec.message
            for rec in caplog.records
        )

    def test_warning_only_logged_once_per_path(self, tmp_path: Path, caplog):
        model_dir = _build_model_dir(
            tmp_path, name="orp2", has_vision_config=True, has_vision_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
        warnings = [
            rec for rec in caplog.records
            if "vision_tower weights missing" in rec.message
        ]
        assert len(warnings) == 1

    def test_load_config_restored_on_normal_exit(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r1", has_vision_config=True, has_vision_weights=False,
        )
        with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
            assert _vu.load_config is not original
        assert _vu.load_config is original

    def test_load_config_restored_on_exception(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r2", has_vision_config=True, has_vision_weights=False,
        )
        with pytest.raises(RuntimeError, match="boom"):
            with _strip_diffusion_gemma_vision_config_if_orphaned(model_dir):
                raise RuntimeError("boom")
        assert _vu.load_config is original

    def test_skips_when_path_is_not_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent-repo"
        sentinel = {
            "model_type": "diffusion_gemma",
            "vision_config": {"hidden_size": 99},
        }
        with patch.object(_vu, "load_config", return_value=sentinel):
            with _strip_diffusion_gemma_vision_config_if_orphaned(nonexistent):
                cfg = _vu.load_config(nonexistent)
        # cfg returned unchanged — vision_config still a dict, not None.
        assert cfg["vision_config"] == {"hidden_size": 99}
