"""Tests for the HuggingFace GGUF client (monkeypatched httpx, no network)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from backend.models import huggingface as hf


def _resp(json_data: Any = None, *, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    req = httpx.Request("GET", "https://huggingface.co/x")
    return httpx.Response(status, json=json_data if json_data is not None else [], request=req, headers=headers or {})


def test_search_gguf_params_and_normalization() -> None:
    async def main() -> None:
        async with hf.HuggingFaceClient() as c:
            captured: dict[str, Any] = {}

            async def fake_get(url: str, params: dict[str, Any] | None = None) -> httpx.Response:
                captured["params"] = params or {}
                return _resp(
                    [{"id": "unsloth/Qwen3-8B-GGUF", "downloads": 100, "likes": 5, "lastModified": "2025"}]
                )

            c._client.get = fake_get  # type: ignore[assignment]
            rows = await c.search_gguf("qwen3")
            assert captured["params"]["filter"] == "gguf"
            assert captured["params"]["search"] == "qwen3"
            assert rows[0]["repo_id"] == "unsloth/Qwen3-8B-GGUF"
            assert rows[0]["downloads"] == 100

    asyncio.run(main())


def test_list_gguf_files_groups_shards_and_parses_quant() -> None:
    async def main() -> None:
        async with hf.HuggingFaceClient() as c:
            tree = [
                {"type": "file", "path": "Qwen3-8B-Q4_K_M.gguf", "lfs": {"size": 4900000000}},
                {"type": "file", "path": "big-00001-of-00002.gguf", "lfs": {"size": 1000}},
                {"type": "file", "path": "big-00002-of-00002.gguf", "lfs": {"size": 2000}},
                {"type": "file", "path": "README.md", "size": 10},
                {"type": "directory", "path": "sub"},
            ]

            async def fake_get(url: str, params: dict[str, Any] | None = None) -> httpx.Response:
                return _resp(tree)

            c._client.get = fake_get  # type: ignore[assignment]
            files = await c.list_gguf_files("r/x")
            by = {f["filename"]: f for f in files}
            assert by["Qwen3-8B-Q4_K_M.gguf"]["quant"] == "Q4_K_M"
            assert by["Qwen3-8B-Q4_K_M.gguf"]["size_bytes"] == 4900000000
            big = by["big-00001-of-00002.gguf"]
            assert big["shards"] == 2
            assert big["size_bytes"] == 3000  # summed across shards
            assert "README.md" not in by

    asyncio.run(main())


def test_fetch_model_config_404_returns_none() -> None:
    async def main() -> None:
        async with hf.HuggingFaceClient() as c:
            async def fake_get(url: str, params: dict[str, Any] | None = None) -> httpx.Response:
                return _resp({}, status=404)

            c._client.get = fake_get  # type: ignore[assignment]
            assert await c.fetch_model_config("r/x") is None

    asyncio.run(main())


def test_module_helpers_tolerant_on_error() -> None:
    class Boom:
        async def search_gguf(self, *_a, **_k):
            raise RuntimeError("down")

        async def list_gguf_files(self, *_a, **_k):
            raise RuntimeError("down")

        async def aclose(self):
            pass

    assert asyncio.run(hf.search("unique-query-zzz", client=Boom())) == []
    assert asyncio.run(hf.list_files("r/x", client=Boom())) == []


class _FakeStream:
    def __init__(self, chunks: list[bytes], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    async def aiter_bytes(self, _chunk_size: int = 1):
        for c in self._chunks:
            yield c

    async def aread(self) -> bytes:
        return b""


def test_stream_download_writes_atomically(tmp_path) -> None:
    async def main() -> None:
        async with hf.HuggingFaceClient() as c:
            def fake_stream(method: str, url: str, headers: dict[str, str] | None = None) -> _FakeStream:
                return _FakeStream([b"abc", b"def"], status_code=200, headers={"Content-Length": "6"})

            c._client.stream = fake_stream  # type: ignore[assignment]
            seen: list[tuple[int, int]] = []

            async def prog(d: int, t: int) -> None:
                seen.append((d, t))

            dest = await c.stream_download("r/x", "model.gguf", tmp_path, progress=prog)
            assert dest.name == "model.gguf"
            assert dest.read_bytes() == b"abcdef"
            assert not (tmp_path / "model.gguf.part").exists()  # atomically renamed
            assert seen[-1] == (6, 6)

    asyncio.run(main())


def test_stream_download_resumes_from_part(tmp_path) -> None:
    async def main() -> None:
        (tmp_path / "model.gguf.part").write_bytes(b"abc")
        async with hf.HuggingFaceClient() as c:
            captured: dict[str, Any] = {}

            def fake_stream(method: str, url: str, headers: dict[str, str] | None = None) -> _FakeStream:
                captured["headers"] = headers
                return _FakeStream([b"def"], status_code=206, headers={"Content-Range": "bytes 3-5/6"})

            c._client.stream = fake_stream  # type: ignore[assignment]
            dest = await c.stream_download("r/x", "model.gguf", tmp_path)
            assert captured["headers"] == {"Range": "bytes=3-"}
            assert dest.read_bytes() == b"abcdef"

    asyncio.run(main())
