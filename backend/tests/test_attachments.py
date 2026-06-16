"""Tests for build_user_content — turning uploaded files into Pydantic AI prompt content.

Pure function, no DB/network/model needed.
"""

from __future__ import annotations

import base64

from pydantic_ai.messages import BinaryContent

from backend.main import build_user_content

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeimage").decode()
PDF = base64.b64encode(b"%PDF-1.4 fake").decode()
HTML = base64.b64encode(b"<html><body><h1>Hi</h1><p>There</p></body></html>").decode()
TXT = base64.b64encode("plain notes".encode()).decode()


def _img(mime: str, data: str, name: str = "f") -> dict:
    return {"filename": name, "mime_type": mime, "data": data}


def test_no_attachments_returns_plain_prompt():
    assert build_user_content("hello", []) == "hello"


def test_image_becomes_binary_content():
    content = build_user_content("what is this", [_img("image/png", PNG, "pic.png")])
    assert isinstance(content, list)
    text_parts = [c for c in content if isinstance(c, str)]
    binaries = [c for c in content if isinstance(c, BinaryContent)]
    assert text_parts and "what is this" in text_parts[0]
    assert len(binaries) == 1
    assert binaries[0].media_type == "image/png"


def test_pdf_becomes_binary_content():
    content = build_user_content("read it", [_img("application/pdf", PDF, "doc.pdf")])
    binaries = [c for c in content if isinstance(c, BinaryContent)]
    assert len(binaries) == 1 and binaries[0].media_type == "application/pdf"


def test_html_is_extracted_and_inlined_as_text():
    content = build_user_content("", [_img("text/html", HTML, "page.html")])
    assert isinstance(content, list)
    joined = "\n".join(c for c in content if isinstance(c, str))
    assert "page.html" in joined
    assert "Hi" in joined and "There" in joined
    assert "<html>" not in joined  # tags stripped by html_to_text
    assert not [c for c in content if isinstance(c, BinaryContent)]


def test_plain_text_inlined():
    content = build_user_content("summarize", [_img("text/plain", TXT, "notes.txt")])
    joined = "\n".join(c for c in content if isinstance(c, str))
    assert "plain notes" in joined and "notes.txt" in joined


def test_unknown_mime_is_skipped():
    # An unsupported type with no other content falls back to the plain prompt.
    content = build_user_content("hi", [_img("application/zip", PNG, "a.zip")])
    assert content == "hi"


def test_empty_prompt_with_only_image_synthesizes_prompt():
    content = build_user_content("", [_img("image/jpeg", PNG)])
    text_parts = [c for c in content if isinstance(c, str)]
    assert text_parts and "attached" in text_parts[0].lower()


def test_non_vision_note_prepended_for_binary():
    content = build_user_content("look", [_img("image/png", PNG)], vision=False)
    note = next(c for c in content if isinstance(c, str))
    assert "may not be able to view" in note
    # No note when the model is vision-capable.
    vis = build_user_content("look", [_img("image/png", PNG)], vision=True)
    assert "may not be able to view" not in next(c for c in vis if isinstance(c, str))


def test_long_text_is_truncated():
    big = base64.b64encode(("x" * 300_000).encode()).decode()
    content = build_user_content("", [_img("text/plain", big, "big.txt")])
    joined = "\n".join(c for c in content if isinstance(c, str))
    assert "…[truncated]" in joined


def test_bad_base64_is_skipped_not_raised():
    content = build_user_content("hi", [_img("image/png", "!!!not base64!!!")])
    # The undecodable image is dropped; the text prompt survives.
    assert content == "hi" or (isinstance(content, list) and "hi" in content[0])
