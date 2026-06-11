"""Tests for block-diffusion generation routing in VLMBatchedEngine.

Background: DiffusionGemma (model_type: diffusion_gemma) uses a masked
block-diffusion loop, not autoregressive next-token prediction. When the engine
detects _is_diffusion=True it must bypass the scheduler and call
mlx_vlm.generate.stream_generate instead, mapping GenerationResult fields to
GenerationOutput.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.vlm import VLMBatchedEngine
from omlx.engine.base import GenerationOutput


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeGenerationResult:
    text: str = "Strength, courage, and kindness."
    prompt_tokens: int = 12
    generation_tokens: int = 7
    finish_reason: str = "stop"
    cached_tokens: int = 0


def _make_engine(is_diffusion: bool = True) -> VLMBatchedEngine:
    """Build a VLMBatchedEngine with minimal fakes for unit testing."""
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_name = "~/.omlx/models/google/diffusiongemma-26B-A4B-it"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    engine._stream_interval = 1
    engine._enable_thinking = None
    engine._model_settings = None
    engine._prefill_eviction_callback = None
    engine._vlm_model = MagicMock(name="vlm_model")
    engine._processor = MagicMock(name="processor")
    engine._tokenizer = MagicMock(name="tokenizer")
    engine._adapter = None
    engine._loaded = True
    engine._grammar_compiler = None
    engine._grammar_compiler_init_attempted = False
    engine._vision_cache = None
    engine._vision_cache_enabled = True
    engine._vlm_mtp_drafter = None
    engine._is_diffusion = is_diffusion
    engine._ocr_stop_ids_cache = None

    # Minimal inner engine mock with an _mlx_executor
    inner = MagicMock(name="inner_engine")
    inner._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine = inner

    return engine


# ---------------------------------------------------------------------------
# _run_diffusion_generate
# ---------------------------------------------------------------------------


class TestRunDiffusionGenerate:
    @pytest.mark.asyncio
    async def test_maps_generation_result_to_generation_output(self):
        engine = _make_engine()
        fake_result = _FakeGenerationResult()

        with patch(
            "mlx_vlm.generate.stream_generate",
            return_value=[fake_result],
        ):
            output = await engine._run_diffusion_generate(
                prompt="What is good in life?",
                max_tokens=64,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
            )

        assert output.text == "Strength, courage, and kindness."
        assert output.prompt_tokens == 12
        assert output.completion_tokens == 7
        assert output.finish_reason == "stop"
        assert output.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_decodes_token_list_prompt(self):
        engine = _make_engine()
        engine._tokenizer.decode.return_value = "What is good in life?"
        fake_result = _FakeGenerationResult()

        captured_prompt = {}

        def _fake_stream_generate(model, processor, prompt, **kwargs):
            captured_prompt["value"] = prompt
            return [fake_result]

        with patch("mlx_vlm.generate.stream_generate", side_effect=_fake_stream_generate):
            await engine._run_diffusion_generate(
                prompt=[1, 2, 3, 4],
                max_tokens=64,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
            )

        engine._tokenizer.decode.assert_called_once_with([1, 2, 3, 4])
        assert captured_prompt["value"] == "What is good in life?"

    @pytest.mark.asyncio
    async def test_returns_empty_output_when_stream_generate_yields_nothing(self):
        engine = _make_engine()

        with patch("mlx_vlm.generate.stream_generate", return_value=[]):
            output = await engine._run_diffusion_generate(
                prompt="Hello?",
                max_tokens=64,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
            )

        assert output.text == ""
        assert output.finish_reason == "length"
        assert output.prompt_tokens == 0
        assert output.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_passes_sampling_params_to_stream_generate(self):
        engine = _make_engine()
        fake_result = _FakeGenerationResult()
        captured_kwargs: dict = {}

        def _fake(model, processor, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return [fake_result]

        with patch("mlx_vlm.generate.stream_generate", side_effect=_fake):
            await engine._run_diffusion_generate(
                prompt="Hello",
                max_tokens=128,
                temperature=0.5,
                top_p=0.85,
                top_k=40,
            )

        assert captured_kwargs["max_tokens"] == 128
        assert captured_kwargs["temperature"] == 0.5
        assert captured_kwargs["top_p"] == 0.85
        assert captured_kwargs["top_k"] == 40

    @pytest.mark.asyncio
    async def test_uses_fallback_finish_reason_when_missing(self):
        engine = _make_engine()
        fake_result = _FakeGenerationResult(finish_reason=None)

        with patch("mlx_vlm.generate.stream_generate", return_value=[fake_result]):
            output = await engine._run_diffusion_generate(
                prompt="Hi",
                max_tokens=32,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
            )

        assert output.finish_reason == "stop"


# ---------------------------------------------------------------------------
# generate() bypass
# ---------------------------------------------------------------------------


class TestGenerateDiffusionBypass:
    @pytest.mark.asyncio
    async def test_generate_calls_diffusion_helper_when_flag_set(self):
        engine = _make_engine(is_diffusion=True)
        expected = GenerationOutput(
            text="Victory.", prompt_tokens=5, completion_tokens=2, finish_reason="stop"
        )
        engine._run_diffusion_generate = AsyncMock(return_value=expected)

        output = await engine.generate(
            prompt="What is good in life?",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        )

        engine._run_diffusion_generate.assert_awaited_once()
        assert output.text == "Victory."

    @pytest.mark.asyncio
    async def test_generate_skips_diffusion_helper_when_flag_false(self):
        engine = _make_engine(is_diffusion=False)
        engine._run_diffusion_generate = AsyncMock()

        # Scheduler path — patch the inner engine.generate to avoid real work.
        from omlx.request import SamplingParams

        inner_result = MagicMock()
        inner_result.output_text = "autoregressive response"
        inner_result.prompt_tokens = 5
        inner_result.completion_tokens = 3
        inner_result.finish_reason = "stop"
        inner_result.tool_calls = None
        inner_result.cached_tokens = 0
        engine._engine.generate = AsyncMock(return_value=inner_result)

        output = await engine.generate(
            prompt="Hello",
            max_tokens=32,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        )

        engine._run_diffusion_generate.assert_not_awaited()
        assert output.text == "autoregressive response"


# ---------------------------------------------------------------------------
# stream_generate() bypass
# ---------------------------------------------------------------------------


def _make_iter_mock(*results):
    """Return an async-generator callable that yields the given GenerationResults.

    Assigned directly to an instance attribute so it is NOT a bound method;
    Python does not prepend self when calling instance-dict functions.
    """
    async def _iter(prompt_str, max_tokens, temperature, top_p, top_k):
        for r in results:
            yield r
    return _iter


class TestStreamGenerateDiffusionBypass:
    @pytest.mark.asyncio
    async def test_stream_generate_streams_incremental_chunks(self):
        engine = _make_engine(is_diffusion=True)

        # Simulate: empty draft marker, two text segments, final segment w/ finish_reason.
        r_draft = _FakeGenerationResult(text="", generation_tokens=0)
        r1 = _FakeGenerationResult(text="Strength", generation_tokens=1, finish_reason=None)
        r2 = _FakeGenerationResult(text=" and courage.", generation_tokens=3, finish_reason="stop")

        engine._iter_diffusion_results = _make_iter_mock(r_draft, r1, r2)

        chunks = []
        async for chunk in engine.stream_generate(
            prompt="What is good?",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        ):
            chunks.append(chunk)

        # Empty draft marker skipped; 2 real chunks yielded.
        assert len(chunks) == 2
        # First chunk: incremental only.
        assert chunks[0].new_text == "Strength"
        assert chunks[0].text == "Strength"
        assert chunks[0].finished is False
        assert chunks[0].finish_reason is None
        # Second (final) chunk: accumulated text, finished.
        assert chunks[1].new_text == "and courage."  # leading space stripped by clean_special_tokens
        assert chunks[1].text == "Strength and courage."
        assert chunks[1].finished is True
        assert chunks[1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_generate_does_not_enter_diffusion_path_when_flag_false(self):
        engine = _make_engine(is_diffusion=False)
        engine._run_diffusion_generate = AsyncMock()

        # Wire a minimal scheduler stream so the normal path doesn't crash.
        async def _fake_stream(request_id):
            out = MagicMock()
            out.output_text = "hi"
            out.new_text = "hi"
            out.prompt_tokens = 2
            out.completion_tokens = 1
            out.finished = True
            out.finish_reason = "stop"
            out.tool_calls = None
            out.cached_tokens = 0
            yield out

        engine._engine.add_request = AsyncMock(return_value="req-1")
        engine._engine.stream_outputs = _fake_stream

        chunks = []
        async for chunk in engine.stream_generate(
            prompt="Hello",
            max_tokens=16,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        ):
            chunks.append(chunk)

        engine._run_diffusion_generate.assert_not_awaited()
        assert len(chunks) >= 1
