"""Tests for the audio_tower fallback and key-remap logic in VLM loading.

Background: oQ-quantized multimodal Gemma 4 checkpoints sometimes ship with
`audio_config` in `config.json` but no `audio_tower.*` weights in the
safetensors. Loading them via `mlx_vlm.utils.load(...)` then crashes with
"Missing 752 parameters" because mlx-vlm instantiates `AudioEncoder` based
on `audio_config`. The `_strip_audio_config_if_orphaned` context manager
swaps `mlx_vlm.utils.load_config` for the duration of the call so that the
config is read with `audio_config = None` when audio weights are absent,
letting the model load without audio support.

Additionally, MLX-format checkpoints skip ``Model.sanitize``, so key remaps
that sanitize normally applies must be re-applied during ``load_weights``.
For ``gemma4_unified``, this means stripping the outer ``model.`` prefix and
inserting ``model.`` inside ``language_model.*`` sub-paths.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import mlx_vlm.utils as _vu

from omlx.engine.vlm import (
    _AUDIO_CONFIG_KEYS,
    _NESTED_VIS_PREFIX,
    _VISION_TOWER_PREFIX,
    _apply_gemma4_unified_key_remap,
    _has_audio_weights,
    _remap_nested_visual_on_load,
    _strip_audio_config_if_orphaned,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """Write a tiny safetensors file with the given parameter keys."""
    from safetensors.numpy import save_file
    import numpy as np

    payload = {k: np.zeros((1,), dtype=np.float32) for k in keys}
    save_file(payload, str(path))


def _build_model_dir(
    tmp_path: Path,
    *,
    name: str,
    has_audio_config: bool,
    has_audio_weights: bool,
) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()

    config: dict = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {"hidden_size": 32, "num_hidden_layers": 1},
        "vision_config": {"hidden_size": 16},
    }
    if has_audio_config:
        config["audio_config"] = {"hidden_size": 16}
        config["audio_token_id"] = 258881
        config["boa_token_id"] = 256000
        config["eoa_token_id"] = 258883
        config["eoa_token_index"] = 258883
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    if has_audio_weights:
        keys.append("audio_tower.layers.0.feed_forward1.linear.weight")
        keys.append("embed_audio.embedding_projection.weight")
    _write_safetensors(model_dir / "model.safetensors", keys)

    return model_dir


# ---------------------------------------------------------------------------
# _has_audio_weights
# ---------------------------------------------------------------------------


class TestHasAudioWeights:
    def test_returns_true_when_audio_tower_key_present(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m1", has_audio_config=True, has_audio_weights=True,
        )
        assert _has_audio_weights(model_dir) is True

    def test_returns_false_when_no_audio_keys(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m2", has_audio_config=True, has_audio_weights=False,
        )
        assert _has_audio_weights(model_dir) is False

    def test_returns_false_for_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _has_audio_weights(empty) is False


# ---------------------------------------------------------------------------
# _strip_audio_config_if_orphaned
# ---------------------------------------------------------------------------


class TestStripAudioConfigIfOrphaned:
    def test_passthrough_when_config_has_no_audio(self, tmp_path: Path):
        # Config with no audio_config — patch must leave the dict untouched.
        model_dir = _build_model_dir(
            tmp_path, name="vision_only",
            has_audio_config=False, has_audio_weights=False,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert "audio_config" not in cfg

    def test_passthrough_when_audio_weights_present(self, tmp_path: Path):
        # Healthy multimodal model — audio_config must remain in the dict.
        model_dir = _build_model_dir(
            tmp_path, name="full",
            has_audio_config=True, has_audio_weights=True,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert cfg.get("audio_config") is not None

    def test_strips_audio_when_weights_missing(self, tmp_path: Path, caplog):
        # Defective oQ-style checkpoint: audio_config present, audio weights absent.
        model_dir = _build_model_dir(
            tmp_path, name="defective",
            has_audio_config=True, has_audio_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_audio_config_if_orphaned(model_dir):
                cfg = _vu.load_config(model_dir)
        # audio_config must be explicitly None (not popped) so mlx-vlm's
        # `setdefault("audio_config", {})` does not repopulate it.
        assert "audio_config" in cfg
        assert cfg["audio_config"] is None
        # Other audio-related keys are popped.
        for k in _AUDIO_CONFIG_KEYS:
            if k != "audio_config":
                assert k not in cfg
        # WARN log fired.
        assert any(
            "audio_tower weights missing" in rec.message
            for rec in caplog.records
        )

    def test_warning_only_logged_once_per_path(self, tmp_path: Path, caplog):
        model_dir = _build_model_dir(
            tmp_path, name="def2",
            has_audio_config=True, has_audio_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_audio_config_if_orphaned(model_dir):
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
        warnings = [
            rec for rec in caplog.records
            if "audio_tower weights missing" in rec.message
        ]
        assert len(warnings) == 1

    def test_load_config_restored_on_normal_exit(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r1",
            has_audio_config=True, has_audio_weights=False,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            assert _vu.load_config is not original
        assert _vu.load_config is original

    def test_load_config_restored_on_exception(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r2",
            has_audio_config=True, has_audio_weights=False,
        )
        with pytest.raises(RuntimeError, match="boom"):
            with _strip_audio_config_if_orphaned(model_dir):
                raise RuntimeError("boom")
        assert _vu.load_config is original

    def test_skips_when_path_is_not_directory(self, tmp_path: Path):
        # When the patched loader is called with a non-directory path (e.g.
        # an HF repo ID before download), the audio_config branch must defer
        # to mlx-vlm's normal flow rather than error out.
        nonexistent = tmp_path / "nonexistent-repo"
        sentinel = {
            "audio_config": {"hidden_size": 99},
            "audio_token_id": 12345,
        }
        with patch.object(_vu, "load_config", return_value=sentinel):
            with _strip_audio_config_if_orphaned(nonexistent):
                cfg = _vu.load_config(nonexistent)
        # cfg returned unchanged — audio_config still a dict, not None.
        assert cfg["audio_config"] == {"hidden_size": 99}
        assert cfg["audio_token_id"] == 12345


# ---------------------------------------------------------------------------
# _has_audio_weights — model.-prefixed keys (gemma4_unified MLX checkpoints)
# ---------------------------------------------------------------------------


class TestHasAudioWeightsModelPrefix:
    """MLX-format gemma4_unified checkpoints store audio weights as
    ``model.embed_audio.*`` (pre-sanitize HF key format).  The detector must
    recognise this prefix so audio_config is not incorrectly stripped."""

    def _write_safetensors(self, path: Path, keys: list) -> None:
        from safetensors.numpy import save_file
        import numpy as np
        save_file({k: np.zeros((1,), dtype=np.float32) for k in keys}, str(path))

    def test_detects_model_embed_audio_prefix(self, tmp_path: Path):
        d = tmp_path / "m"
        d.mkdir()
        self._write_safetensors(
            d / "model.safetensors",
            ["model.embed_audio.embedding_projection.weight",
             "model.language_model.layers.0.self_attn.q_proj.weight"],
        )
        assert _has_audio_weights(d) is True

    def test_detects_model_audio_tower_prefix(self, tmp_path: Path):
        d = tmp_path / "m2"
        d.mkdir()
        self._write_safetensors(
            d / "model.safetensors",
            ["model.audio_tower.layers.0.weight"],
        )
        assert _has_audio_weights(d) is True

    def test_no_false_positive_for_non_audio_model_prefix(self, tmp_path: Path):
        d = tmp_path / "m3"
        d.mkdir()
        self._write_safetensors(
            d / "model.safetensors",
            ["model.language_model.layers.0.self_attn.q_proj.weight",
             "model.embed_vision.embedding_projection.weight"],
        )
        assert _has_audio_weights(d) is False


# ---------------------------------------------------------------------------
# _apply_gemma4_unified_key_remap — unit tests for the key transform helper
# ---------------------------------------------------------------------------


class TestApplyGemma4UnifiedKeyRemap:
    """The helper reproduces what ``gemma4_unified.Model.sanitize`` does for
    each key: strip outer ``model.`` prefix and insert ``model.`` inside
    language_model sub-paths that lack it."""

    def test_language_model_layer_key(self):
        k = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert _apply_gemma4_unified_key_remap(k) == (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        )

    def test_embed_tokens_key(self):
        k = "model.language_model.embed_tokens.weight"
        assert _apply_gemma4_unified_key_remap(k) == "language_model.model.embed_tokens.weight"

    def test_embed_audio_key(self):
        k = "model.embed_audio.embedding_projection.weight"
        assert _apply_gemma4_unified_key_remap(k) == "embed_audio.embedding_projection.weight"

    def test_embed_vision_key(self):
        k = "model.embed_vision.embedding_projection.weight"
        assert _apply_gemma4_unified_key_remap(k) == "embed_vision.embedding_projection.weight"

    def test_already_sanitized_language_model_key_unchanged(self):
        # Keys that already have "language_model.model." must not be double-remapped.
        k = "language_model.model.layers.0.self_attn.q_proj.weight"
        assert _apply_gemma4_unified_key_remap(k) == k

    def test_already_sanitized_embed_key_unchanged(self):
        k = "embed_audio.embedding_projection.weight"
        assert _apply_gemma4_unified_key_remap(k) == k

    def test_key_without_model_prefix_unchanged(self):
        k = "lm_head.weight"
        assert _apply_gemma4_unified_key_remap(k) == k


# ---------------------------------------------------------------------------
# _remap_nested_visual_on_load — integration: context manager applies remap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _strip_audio_config_if_orphaned — gemma4_unified quantization key remap
# ---------------------------------------------------------------------------


class TestStripAudioConfigQuantKeyRemap:
    """Per-layer quantization dict keys use HF format (``model.language_model.*``)
    in oQ4 checkpoints.  nn.quantize's class_predicate receives model-tree paths
    (``language_model.model.*``), so the keys must be remapped before the config
    reaches load_model for gemma4_unified."""

    def _make_dir(self, tmp_path, quant_dict, model_type="gemma4_unified"):
        d = tmp_path / "gu"
        d.mkdir()
        cfg = {
            "model_type": model_type,
            "quantization": dict(quant_dict),
            "quantization_config": dict(quant_dict),
        }
        (d / "config.json").write_text(json.dumps(cfg))
        from safetensors.numpy import save_file
        import numpy as np
        save_file({"k": np.zeros((1,), dtype=np.float32)}, str(d / "model.safetensors"))
        return d

    def test_per_layer_keys_remapped_in_quantization(self, tmp_path):
        raw_quant = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.language_model.layers.0.mlp.down_proj": {
                "bits": 6, "group_size": 64, "mode": "affine",
            },
            "model.language_model.layers.0.self_attn.q_proj": {
                "bits": 5, "group_size": 64, "mode": "affine",
            },
            "model.embed_vision.embedding_projection": {
                "bits": 4, "group_size": 64, "mode": "affine",
            },
        }
        d = self._make_dir(tmp_path, raw_quant)
        with _strip_audio_config_if_orphaned(d):
            cfg = _vu.load_config(d)
        quant = cfg["quantization"]
        # HF-format keys must be replaced by model-tree paths.
        assert "model.language_model.layers.0.mlp.down_proj" not in quant
        assert "model.language_model.layers.0.self_attn.q_proj" not in quant
        assert "model.embed_vision.embedding_projection" not in quant
        assert "language_model.model.layers.0.mlp.down_proj" in quant
        assert quant["language_model.model.layers.0.mlp.down_proj"]["bits"] == 6
        assert "language_model.model.layers.0.self_attn.q_proj" in quant
        assert quant["language_model.model.layers.0.self_attn.q_proj"]["bits"] == 5
        assert "embed_vision.embedding_projection" in quant
        # Scalar top-level keys (group_size, bits, mode) are preserved.
        assert quant["group_size"] == 64
        assert quant["bits"] == 4

    def test_per_layer_keys_remapped_in_quantization_config(self, tmp_path):
        raw_quant = {
            "group_size": 64,
            "bits": 4,
            "model.language_model.layers.1.mlp.gate_proj": {
                "bits": 6, "group_size": 64, "mode": "affine",
            },
        }
        d = self._make_dir(tmp_path, raw_quant)
        with _strip_audio_config_if_orphaned(d):
            cfg = _vu.load_config(d)
        qc = cfg["quantization_config"]
        assert "model.language_model.layers.1.mlp.gate_proj" not in qc
        assert "language_model.model.layers.1.mlp.gate_proj" in qc

    def test_scalar_values_not_remapped(self, tmp_path):
        raw_quant = {"group_size": 64, "bits": 4, "mode": "affine"}
        d = self._make_dir(tmp_path, raw_quant)
        with _strip_audio_config_if_orphaned(d):
            cfg = _vu.load_config(d)
        assert cfg["quantization"]["group_size"] == 64
        assert cfg["quantization"]["bits"] == 4

    def test_non_gemma4_unified_keys_not_remapped(self, tmp_path):
        """Other model types (e.g. gemma4) must not have their quant keys touched."""
        raw_quant = {
            "group_size": 64,
            "bits": 4,
            "model.language_model.layers.0.mlp.down_proj": {
                "bits": 6, "group_size": 64,
            },
        }
        d = self._make_dir(tmp_path, raw_quant, model_type="gemma4")
        with _strip_audio_config_if_orphaned(d):
            cfg = _vu.load_config(d)
        quant = cfg["quantization"]
        assert "model.language_model.layers.0.mlp.down_proj" in quant
        assert "language_model.model.layers.0.mlp.down_proj" not in quant


class TestRemapNestedVisualOnLoad:
    """Integration tests for the context manager.  We install a fake
    ``_vu.load_model`` (as the "original") and verify that calling the patched
    version routes weight keys through the right transforms."""

    def _make_model_dir(self, tmp_path: Path, model_type: str) -> Path:
        d = tmp_path / model_type
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
        return d

    def _remap_via_context(
        self, model_dir: Path, raw_keys: list[str], monkeypatch
    ) -> list[str]:
        """Run raw_keys through the context manager's load_weights intercept
        and return the keys that reach the bottom-level load_weights call."""
        import mlx_vlm.utils as _vu2
        import mlx.nn as _nn

        received: list[tuple] = []

        # Bottom-level tracker — captures whatever the context manager passes
        # after remapping.
        orig_lw = _nn.Module.load_weights

        def tracking_lw(self, items, *args, **kw):
            if not isinstance(items, str):
                received.extend(items)
            return orig_lw(self, items if isinstance(items, str) else [], *args, **kw)

        # Fake "original" load_model that just calls load_weights with raw_keys.
        # _remap_nested_visual_on_load saves this as its "original" and wraps it.
        def fake_load_model(model_path, lazy=False, **kwargs):
            class _M(_nn.Module):
                pass
            m = _M()
            m.load_weights([(k, None) for k in raw_keys], strict=False)
            return m, None

        monkeypatch.setattr(_vu2, "load_model", fake_load_model)
        monkeypatch.setattr(_nn.Module, "load_weights", tracking_lw)

        with _remap_nested_visual_on_load(model_dir):
            _vu2.load_model(str(model_dir))

        return [k for k, _ in received]

    def test_gemma4_unified_remap_applied(self, tmp_path: Path, monkeypatch):
        model_dir = self._make_model_dir(tmp_path, "gemma4_unified")
        raw_keys = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.embed_audio.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
        ]
        result = self._remap_via_context(model_dir, raw_keys, monkeypatch)
        assert "language_model.model.layers.0.self_attn.q_proj.weight" in result
        assert "embed_audio.embedding_projection.weight" in result
        assert "embed_vision.embedding_projection.weight" in result
        assert not any(k.startswith("model.") for k in result)

    def test_gemma4_unified_already_sanitized_keys_unchanged(self, tmp_path, monkeypatch):
        model_dir = self._make_model_dir(tmp_path, "gemma4_unified")
        raw_keys = [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "embed_audio.embedding_projection.weight",
        ]
        result = self._remap_via_context(model_dir, raw_keys, monkeypatch)
        assert result == raw_keys

    def test_non_gemma4_model_prefix_keys_untouched(self, tmp_path, monkeypatch):
        model_dir = self._make_model_dir(tmp_path, "qwen2_vl")
        raw_keys = ["model.language_model.layers.0.weight"]
        result = self._remap_via_context(model_dir, raw_keys, monkeypatch)
        assert result == raw_keys

    def test_nested_vis_remap_still_fires_for_non_gemma4(self, tmp_path, monkeypatch):
        model_dir = self._make_model_dir(tmp_path, "qwen2_vl")
        raw_keys = [
            f"{_NESTED_VIS_PREFIX}layer.0.weight",
            "language_model.model.layers.0.weight",
        ]
        result = self._remap_via_context(model_dir, raw_keys, monkeypatch)
        assert f"{_VISION_TOWER_PREFIX}layer.0.weight" in result
        assert "language_model.model.layers.0.weight" in result

    # ------------------------------------------------------------------
    # _load_safetensors pre-remap (fixes nn.quantize class_predicate for
    # MLX-format oQ4 gemma4_unified checkpoints)
    # ------------------------------------------------------------------

    def test_gemma4_unified_load_safetensors_pre_remaps(self, tmp_path, monkeypatch):
        """Within _remap_nested_visual_on_load for gemma4_unified, calls to
        _vu._load_safetensors return remapped keys so nn.quantize's
        class_predicate (``f"{p}.scales" in weights``) works correctly."""
        import mlx_vlm.utils as _vu2

        model_dir = self._make_model_dir(tmp_path, "gemma4_unified")
        captured: list[dict] = []

        def fake_load_st(path):
            return {
                "model.language_model.layers.0.mlp.down_proj.scales": None,
                "model.embed_audio.embedding_projection.scales": None,
                "model.language_model.model.visual.0.weight": None,
            }

        def fake_load_model(model_path, lazy=False, **kwargs):
            import mlx.nn as _nn
            result = _vu2._load_safetensors(str(model_path))
            captured.append(result)
            class _M(_nn.Module):
                pass
            m = _M()
            m.load_weights(list(result.items()), strict=False)
            return m, None

        monkeypatch.setattr(_vu2, "_load_safetensors", fake_load_st)
        monkeypatch.setattr(_vu2, "load_model", fake_load_model)

        with _remap_nested_visual_on_load(model_dir):
            _vu2.load_model(str(model_dir))

        assert len(captured) == 1
        keys = list(captured[0].keys())
        assert "language_model.model.layers.0.mlp.down_proj.scales" in keys
        assert "embed_audio.embedding_projection.scales" in keys
        # model.language_model.model.visual.* → language_model.model.visual.*
        assert "language_model.model.visual.0.weight" in keys
        assert not any(k.startswith("model.") for k in keys)

    def test_non_gemma4_load_safetensors_not_patched(self, tmp_path, monkeypatch):
        """For non-gemma4_unified models, _vu._load_safetensors is not wrapped."""
        import mlx_vlm.utils as _vu2

        model_dir = self._make_model_dir(tmp_path, "gemma4")
        outside_ref = _vu2._load_safetensors
        captured_ref: list = []

        def fake_load_model(model_path, lazy=False, **kwargs):
            import mlx.nn as _nn
            captured_ref.append(_vu2._load_safetensors)
            class _M(_nn.Module):
                pass
            return _M(), None

        monkeypatch.setattr(_vu2, "load_model", fake_load_model)

        with _remap_nested_visual_on_load(model_dir):
            _vu2.load_model(str(model_dir))

        assert len(captured_ref) == 1
        assert captured_ref[0] is outside_ref

    def test_load_safetensors_restored_on_context_exit(self, tmp_path):
        """_vu._load_safetensors is patched inside the context and restored on exit."""
        import mlx_vlm.utils as _vu2

        model_dir = self._make_model_dir(tmp_path, "gemma4_unified")
        original_load_st = _vu2._load_safetensors

        with _remap_nested_visual_on_load(model_dir):
            inside_ref = _vu2._load_safetensors

        assert _vu2._load_safetensors is original_load_st
        assert inside_ref is not original_load_st

    def test_load_safetensors_restored_on_exception(self, tmp_path):
        """_vu._load_safetensors is restored even when the with block raises."""
        import mlx_vlm.utils as _vu2

        model_dir = self._make_model_dir(tmp_path, "gemma4_unified")
        original_load_st = _vu2._load_safetensors

        with pytest.raises(RuntimeError, match="boom"):
            with _remap_nested_visual_on_load(model_dir):
                raise RuntimeError("boom")

        assert _vu2._load_safetensors is original_load_st


class TestPatchVideoProcessorBug:
    """_patch_video_processor_bug filters invalid kwargs from Gemma4UnifiedVideoProcessor."""

    def test_invalid_kwargs_stripped_from_unified_video_processor(self):
        """do_convert_rgb and other image-only kwargs from oQ4 processor_config.json
        must not reach Gemma4VideoProcessor.__init__, which rejects them."""
        from omlx.engine.vlm import _patch_video_processor_bug

        _patch_video_processor_bug()

        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedVideoProcessor,
        )

        # These are the fields present in oQ4 checkpoint processor_config.json
        # that Gemma4VideoProcessor.__init__ does not accept.
        vp = Gemma4UnifiedVideoProcessor(
            patch_size=16,
            max_soft_tokens=70,
            pooling_kernel_size=3,
            num_frames=32,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            # Invalid args that should be silently dropped:
            do_convert_rgb=True,
            do_resize=True,
            do_sample_frames=True,
            resample=3,
            return_metadata=False,
        )
        assert vp.patch_size == 16
        assert vp.num_frames == 32

    def test_valid_kwargs_still_applied(self):
        """Valid kwargs are not accidentally stripped."""
        from omlx.engine.vlm import _patch_video_processor_bug

        _patch_video_processor_bug()

        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedVideoProcessor,
        )

        vp = Gemma4UnifiedVideoProcessor(
            patch_size=32,
            num_frames=16,
            do_normalize=False,
        )
        assert vp.patch_size == 32
        assert vp.num_frames == 16
        assert vp.do_normalize is False
